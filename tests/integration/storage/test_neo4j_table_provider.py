# Copyright (C) 2026 Microsoft
# Licensed under the MIT License

"""Integration coverage for the Neo4j table provider."""

import os
import uuid

import pandas as pd
import pytest
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider

pytestmark = pytest.mark.skipif(
    os.getenv("GRAPHRAG_TEST_NEO4J") != "1",
    reason="requires a running Neo4j instance and GRAPHRAG_TEST_NEO4J=1",
)


@pytest.fixture
async def provider():
    namespace = f"test-{uuid.uuid4()}"
    result = Neo4jTableProvider(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:17687"),
        password=os.environ["NEO4J_PASSWORD"],
        namespace=namespace,
    )
    try:
        yield result
    finally:
        async with result._driver.session(database="neo4j") as session:  # noqa: SLF001
            await session.run(
                "MATCH (node) "
                "WHERE node.namespace = $namespace "
                "OR node.namespace STARTS WITH $namespace_prefix "
                "OR node.logical_namespace = $namespace "
                "DETACH DELETE node",
                namespace=namespace,
                namespace_prefix=f"{namespace}/",
            )
        await result.close()


@pytest.mark.asyncio
async def test_round_trip_and_graph_projection(provider: Neo4jTableProvider):
    tables = {
        "entities": pd.DataFrame([
            {"id": "entity-a", "title": "Alice", "description": "Person"},
            {"id": "entity-b", "title": "Bob", "description": "Person"},
        ]),
        "relationships": pd.DataFrame([
            {
                "id": "relationship-1",
                "source": "Alice",
                "target": "Bob",
                "description": "knows",
                "weight": 1.0,
                "text_unit_ids": ["text-1"],
            }
        ]),
        "text_units": pd.DataFrame([
            {
                "id": "text-1",
                "text": "Alice knows Bob.",
                "document_id": "document-1",
                "entity_ids": ["entity-a", "entity-b"],
                "relationship_ids": ["relationship-1"],
            }
        ]),
        "documents": pd.DataFrame([
            {
                "id": "document-1",
                "title": "Example",
                "text": "Alice knows Bob.",
                "text_unit_ids": ["text-1"],
            }
        ]),
        "communities": pd.DataFrame([
            {
                "id": "community-1",
                "title": "People",
                "entity_ids": ["entity-a", "entity-b"],
                "relationship_ids": ["relationship-1"],
                "text_unit_ids": ["text-1"],
                "children": [],
            }
        ]),
        "community_reports": pd.DataFrame([
            {
                "id": "report-1",
                "community": "community-1",
                "title": "People report",
                "summary": "Alice knows Bob.",
                "full_content": "Alice knows Bob.",
            }
        ]),
    }

    for table_name, dataframe in tables.items():
        await provider.write_dataframe(table_name, dataframe)
        pd.testing.assert_frame_equal(
            await provider.read_dataframe(table_name), dataframe
        )

    async with provider._driver.session(database="neo4j") as session:  # noqa: SLF001
        result = await session.run(
            "MATCH (:GraphRAGRecord {namespace: $namespace})-[relationship]->() "
            "RETURN count(relationship) AS count",
            namespace=provider._namespace,  # noqa: SLF001
        )
        record = await result.single()

    assert record is not None
    assert record["count"] == 10
    assert set(provider.list()) == set(tables)
    assert (await provider.get_active_provider()).vector_version is None


@pytest.mark.asyncio
async def test_version_publish_and_failure_keep_active(provider: Neo4jTableProvider):
    first_version = await provider.create_building_version("first")
    await first_version.write_dataframe(
        "entities", pd.DataFrame([{"id": "entity-1", "title": "first"}])
    )
    await provider.publish_version("first")

    active = await provider.get_active_provider()
    assert active.vector_version == "first"
    assert (await active.read_dataframe("entities")).iloc[0]["title"] == "first"

    failed_version = await provider.create_building_version("failed")
    await failed_version.write_dataframe(
        "entities", pd.DataFrame([{"id": "entity-2", "title": "failed"}])
    )
    await provider.mark_version_failed("failed")

    active_after_failure = await provider.get_active_provider()
    assert active_after_failure.vector_version == "first"
    assert (
        await active_after_failure.read_dataframe("entities")
    ).iloc[0]["title"] == "first"

    second_version = await provider.create_building_version("second")
    await second_version.write_dataframe(
        "entities", pd.DataFrame([{"id": "entity-3", "title": "second"}])
    )
    await provider.publish_version("second")

    active_after_publish = await provider.get_active_provider()
    assert active_after_publish.vector_version == "second"
    assert (
        await active_after_publish.read_dataframe("entities")
    ).iloc[0]["title"] == "second"

    await provider.clear_versions(["first", "failed", "second"])
    assert not await first_version.has("entities")
    assert not await failed_version.has("entities")
    assert await active_after_publish.has("entities")