# Copyright (C) 2026 Microsoft
# Licensed under the MIT License

"""Neo4j-backed GraphRAG table provider.

Each GraphRAG table row is stored as a labelled ``GraphRAGRecord`` node.  The
original row is retained as JSON so the ``TableProvider`` round trip remains
lossless for pipeline consumers.  For the six final GraphRAG tables, the same
nodes are also connected with domain relationships, enabling graph-native
inspection without changing workflow read/write behavior.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

import pandas as pd
from neo4j import AsyncDriver, AsyncGraphDatabase, Driver, GraphDatabase

from graphrag_storage.tables.table import RowTransformer, Table
from graphrag_storage.tables.table_provider import TableProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_RECORD_LABELS = {
    "documents": "Document",
    "text_units": "TextUnit",
    "entities": "Entity",
    "relationships": "Relationship",
    "communities": "Community",
    "community_reports": "CommunityReport",
}


def _identity(row: dict[str, Any]) -> dict[str, Any]:
    return row


def _apply_transformer(transformer: RowTransformer, row: dict[str, Any]) -> Any:
    if inspect.isclass(transformer):
        return transformer(**row)
    return transformer(row)


def _row_properties(row: dict[str, Any]) -> dict[str, Any]:
    """Keep queryable scalar fields while retaining the complete JSON row."""
    properties: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) for item in value
        )):
            properties[key] = value
        elif isinstance(value, (list, dict)):
            properties[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return properties


def _records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = json.loads(df.to_json(orient="records", date_format="iso", force_ascii=False))
    return [
        {
            "row_id": str(row.get("id", index)),
            "data": json.dumps(row, ensure_ascii=False, sort_keys=True),
            "ordinal": index,
            "properties": _row_properties(row),
        }
        for index, row in enumerate(rows)
    ]


class Neo4jTableProvider(TableProvider):
    """Store GraphRAG tables in Neo4j while preserving DataFrame semantics.

    ``namespace`` is extended by :meth:`child`, giving incremental index runs
    the same previous/delta isolation as file-backed table providers.
    """

    def __init__(
        self,
        *,
        uri: str = "bolt://localhost:17687",
        username: str = "neo4j",
        password: str | None = None,
        database: str = "neo4j",
        namespace: str = "",
        _driver: AsyncDriver | None = None,
        _list_driver: Driver | None = None,
        **_: Any,
    ) -> None:
        if not password and _driver is None:
            msg = "Neo4jTableProvider requires a 'password' configuration value."
            raise ValueError(msg)
        self._driver = _driver or AsyncGraphDatabase.driver(uri, auth=(username, password))
        self._list_driver = _list_driver or GraphDatabase.driver(
            uri, auth=(username, password)
        )
        self._database = database
        self._namespace = namespace
        self._owns_driver = _driver is None
        self._owns_list_driver = _list_driver is None
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "CREATE CONSTRAINT graphrag_record_key IF NOT EXISTS "
                "FOR (record:GraphRAGRecord) "
                "REQUIRE (record.namespace, record.table_name, record.row_id) IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT graphrag_table_key IF NOT EXISTS "
                "FOR (table:GraphRAGTable) "
                "REQUIRE (table.namespace, table.name) IS UNIQUE"
            )
        self._initialized = True

    async def read_dataframe(self, table_name: str) -> pd.DataFrame:
        """Read a table back into its original DataFrame shape."""
        await self._ensure_schema()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (table:GraphRAGTable {namespace: $namespace, name: $table_name}) "
                "OPTIONAL MATCH (record:GraphRAGRecord {namespace: $namespace, table_name: $table_name}) "
                "RETURN table.columns AS columns, record.data AS data "
                "ORDER BY record.ordinal",
                namespace=self._namespace,
                table_name=table_name,
            )
            records = [record async for record in result]

        if not records:
            msg = f"Table '{table_name}' not found in namespace '{self._namespace}'."
            raise ValueError(msg)
        columns = json.loads(records[0]["columns"])
        rows = [json.loads(record["data"]) for record in records if record["data"]]
        return pd.DataFrame(rows, columns=columns)

    async def write_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Replace one logical table and rebuild its graph projection."""
        await self._ensure_schema()
        records = _records_from_dataframe(df)
        label = _RECORD_LABELS.get(table_name)
        label_clause = f":{label}" if label else ""

        async with self._driver.session(database=self._database) as session:
            await session.run(
                "MERGE (table:GraphRAGTable {namespace: $namespace, name: $table_name}) "
                "SET table.columns = $columns",
                namespace=self._namespace,
                table_name=table_name,
                columns=json.dumps(list(df.columns), ensure_ascii=False),
            )
            await session.run(
                "MATCH (record:GraphRAGRecord {namespace: $namespace, table_name: $table_name}) "
                "DETACH DELETE record",
                namespace=self._namespace,
                table_name=table_name,
            )
            if records:
                await session.run(
                    "UNWIND $records AS row "
                    f"MERGE (record:GraphRAGRecord{label_clause} "
                    "{namespace: $namespace, table_name: $table_name, row_id: row.row_id}) "
                    "SET record += row.properties "
                    "SET record.namespace = $namespace, record.table_name = $table_name, "
                    "record.row_id = row.row_id, record.data = row.data, record.ordinal = row.ordinal",
                    namespace=self._namespace,
                    table_name=table_name,
                    records=records,
                )

        await self._refresh_domain_relationships(table_name, df)

    def read_local_search_context(
        self, seed_entity_ids: list[str]
    ) -> dict[str, pd.DataFrame]:
        """Load graph-filtered records into the local search context."""
        if not seed_entity_ids:
            return {
                name: pd.DataFrame()
                for name in (
                    "entities",
                    "relationships",
                    "text_units",
                    "communities",
                    "community_reports",
                )
            }

        with self._list_driver.session(database=self._database) as session:
            entity_rows = [
                record["data"]
                for record in session.run(
                    "MATCH (seed:Entity {namespace: $namespace}) "
                    "WHERE seed.row_id IN $seed_ids "
                    "OPTIONAL MATCH (seed)-[:RELATED]-(neighbor:Entity {namespace: $namespace}) "
                    "WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) AS nodes "
                    "UNWIND nodes AS node WITH DISTINCT node WHERE node IS NOT NULL "
                    "RETURN node.data AS data",
                    namespace=self._namespace,
                    seed_ids=seed_entity_ids,
                )
            ]
            entity_ids = [str(json.loads(row)["id"]) for row in entity_rows]
            relationship_rows = [
                record["data"]
                for record in session.run(
                    "MATCH (source:Entity {namespace: $namespace})-[edge:RELATED]-"
                    "(target:Entity {namespace: $namespace}) "
                    "WHERE edge.id IS NOT NULL "
                    "AND (source.row_id IN $entity_ids OR target.row_id IN $entity_ids) "
                    "MATCH (relationship:Relationship {namespace: $namespace, row_id: edge.id}) "
                    "RETURN DISTINCT relationship.data AS data",
                    namespace=self._namespace,
                    entity_ids=entity_ids,
                )
            ]
            relationship_ids = [str(json.loads(row)["id"]) for row in relationship_rows]
            text_unit_rows = [
                record["data"]
                for record in session.run(
                    "MATCH (text_unit:TextUnit {namespace: $namespace}) "
                    "OPTIONAL MATCH (text_unit)-[:MENTIONS]->(entity:Entity {namespace: $namespace}) "
                    "OPTIONAL MATCH (text_unit)-[:EVIDENCES]->"
                    "(relationship:Relationship {namespace: $namespace}) "
                    "WITH text_unit, collect(entity.row_id) AS entity_ids, "
                    "collect(relationship.row_id) AS relationship_ids "
                    "WHERE any(id IN entity_ids WHERE id IN $entity_ids) "
                    "OR any(id IN relationship_ids WHERE id IN $relationship_ids) "
                    "RETURN DISTINCT text_unit.data AS data",
                    namespace=self._namespace,
                    entity_ids=entity_ids,
                    relationship_ids=relationship_ids,
                )
            ]
            community_records = list(session.run(
                "MATCH (community:Community {namespace: $namespace})-[:HAS_MEMBER]->"
                "(member:Entity {namespace: $namespace}) "
                "WHERE member.row_id IN $seed_ids "
                "WITH DISTINCT community "
                "OPTIONAL MATCH (report:CommunityReport {namespace: $namespace})-[:REPORTS_ON]->(community) "
                "RETURN community.data AS community_data, report.data AS report_data",
                namespace=self._namespace,
                seed_ids=seed_entity_ids,
            ))
            community_rows = [
                record["community_data"] for record in community_records if record["community_data"]
            ]
            report_rows = [
                record["report_data"] for record in community_records if record["report_data"]
            ]

        def to_dataframe(rows: list[str]) -> pd.DataFrame:
            return pd.DataFrame([json.loads(row) for row in dict.fromkeys(rows)])

        return {
            "entities": to_dataframe(entity_rows),
            "relationships": to_dataframe(relationship_rows),
            "text_units": to_dataframe(text_unit_rows),
            "communities": to_dataframe(community_rows),
            "community_reports": to_dataframe(report_rows),
        }

    async def has(self, table_name: str) -> bool:
        """Check whether the logical table was created."""
        await self._ensure_schema()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (:GraphRAGTable {namespace: $namespace, name: $table_name}) "
                "RETURN count(*) > 0 AS exists",
                namespace=self._namespace,
                table_name=table_name,
            )
            record = await result.single()
        return bool(record and record["exists"])

    def list(self) -> list[str]:
        """Return table names in the current namespace."""
        with self._list_driver.session(database=self._database) as session:
            result = session.run(
                "MATCH (table:GraphRAGTable {namespace: $namespace}) RETURN table.name AS name",
                namespace=self._namespace,
            )
            return [record["name"] for record in result]

    async def clear(self) -> None:
        await self._ensure_schema()
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "MATCH (node {namespace: $namespace}) DETACH DELETE node",
                namespace=self._namespace,
            )

    def open(
        self,
        table_name: str,
        transformer: RowTransformer | None = None,
        truncate: bool = True,
    ) -> Table:
        """Open a buffered stream for a logical table."""
        return Neo4jTable(self, table_name, transformer, truncate)

    def child(self, name: str | None) -> Neo4jTableProvider:
        """Create a provider scoped to a child namespace."""
        if name is None:
            return self
        namespace = f"{self._namespace}/{name}" if self._namespace else name
        return Neo4jTableProvider(
            database=self._database,
            namespace=namespace,
            _driver=self._driver,
            _list_driver=self._list_driver,
        )

    async def close(self) -> None:
        """Close drivers owned by this provider."""
        if self._owns_driver:
            await self._driver.close()
        if self._owns_list_driver:
            self._list_driver.close()

    async def _refresh_domain_relationships(
        self, table_name: str, df: pd.DataFrame
    ) -> None:
        builders = {
            "documents": self._link_documents,
            "text_units": self._link_text_units,
            "relationships": self._link_entity_relationships,
            "communities": self._link_communities,
            "community_reports": self._link_community_reports,
        }
        builder = builders.get(table_name)
        if builder:
            await builder(df)

    async def _run_links(self, query: str, links: list[dict[str, Any]]) -> None:
        if not links:
            return
        async with self._driver.session(database=self._database) as session:
            await session.run(query, namespace=self._namespace, links=links)

    async def _link_documents(self, df: pd.DataFrame) -> None:
        links = [
            {"source": str(row["id"]), "targets": [str(value) for value in row.get("text_unit_ids", [])]}
            for row in df.to_dict("records")
        ]
        await self._run_links(
            "UNWIND $links AS link "
            "MATCH (document:Document {namespace: $namespace, row_id: link.source}) "
            "UNWIND link.targets AS target_id "
            "MATCH (text_unit:TextUnit {namespace: $namespace, row_id: target_id}) "
            "MERGE (document)-[:CONTAINS]->(text_unit)",
            links,
        )

    async def _link_text_units(self, df: pd.DataFrame) -> None:
        rows = df.to_dict("records")
        entity_links = [
            {"source": str(row["id"]), "targets": [str(value) for value in row.get("entity_ids", [])]}
            for row in rows
        ]
        relationship_links = [
            {"source": str(row["id"]), "targets": [str(value) for value in row.get("relationship_ids", [])]}
            for row in rows
        ]
        await self._run_links(
            "UNWIND $links AS link "
            "MATCH (text_unit:TextUnit {namespace: $namespace, row_id: link.source}) "
            "UNWIND link.targets AS target_id "
            "MATCH (entity:Entity {namespace: $namespace, row_id: target_id}) "
            "MERGE (text_unit)-[:MENTIONS]->(entity)",
            entity_links,
        )
        await self._run_links(
            "UNWIND $links AS link "
            "MATCH (text_unit:TextUnit {namespace: $namespace, row_id: link.source}) "
            "UNWIND link.targets AS target_id "
            "MATCH (relationship:Relationship {namespace: $namespace, row_id: target_id}) "
            "MERGE (text_unit)-[:EVIDENCES]->(relationship)",
            relationship_links,
        )

    async def _link_entity_relationships(self, df: pd.DataFrame) -> None:
        if "id" not in df.columns:
            return

        links = [
            {
                "id": str(row["id"]),
                "source": row.get("source"),
                "target": row.get("target"),
                "properties": _row_properties(row),
            }
            for row in df.to_dict("records")
            if row.get("source") is not None and row.get("target") is not None
        ]
        await self._run_links(
            "UNWIND $links AS link "
            "MATCH (source:Entity {namespace: $namespace, title: link.source}) "
            "MATCH (target:Entity {namespace: $namespace, title: link.target}) "
            "MERGE (source)-[edge:RELATED {id: link.id}]->(target) "
            "SET edge += link.properties",
            links,
        )

    async def _link_communities(self, df: pd.DataFrame) -> None:
        rows = df.to_dict("records")
        for relation, field, label in (
            ("HAS_MEMBER", "entity_ids", "Entity"),
            ("INCLUDES_RELATIONSHIP", "relationship_ids", "Relationship"),
            ("CONTAINS_TEXT_UNIT", "text_unit_ids", "TextUnit"),
            ("HAS_CHILD", "children", "Community"),
        ):
            links = [
                {"source": str(row["id"]), "targets": [str(value) for value in row.get(field, [])]}
                for row in rows
            ]
            await self._run_links(
                "UNWIND $links AS link "
                "MATCH (community:Community {namespace: $namespace, row_id: link.source}) "
                "UNWIND link.targets AS target_id "
                f"MATCH (target:{label} {{namespace: $namespace, row_id: target_id}}) "
                f"MERGE (community)-[:{relation}]->(target)",
                links,
            )

    async def _link_community_reports(self, df: pd.DataFrame) -> None:
        links = [
            {"source": str(row["id"]), "target": str(row["community"])}
            for row in df.to_dict("records")
            if row.get("community") is not None
        ]
        await self._run_links(
            "UNWIND $links AS link "
            "MATCH (report:CommunityReport {namespace: $namespace, row_id: link.source}) "
            "MATCH (community:Community {namespace: $namespace}) "
            "WHERE toString(community.community) = link.target "
            "MERGE (report)-[:REPORTS_ON]->(community)",
            links,
        )


class Neo4jTable(Table):
    """Buffered row interface matching the existing Parquet table behavior."""

    def __init__(
        self,
        provider: Neo4jTableProvider,
        table_name: str,
        transformer: RowTransformer | None,
        truncate: bool,
    ) -> None:
        self._provider = provider
        self._table_name = table_name
        self._transformer = transformer or _identity
        self._truncate = truncate
        self._read_df: pd.DataFrame | None = None
        self._write_rows: list[dict[str, Any]] = []

    def __aiter__(self) -> AsyncIterator[Any]:  # noqa: D105
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        if self._read_df is None:
            if await self._provider.has(self._table_name):
                self._read_df = await self._provider.read_dataframe(self._table_name)
            else:
                self._read_df = pd.DataFrame()
        for row in self._read_df.to_dict("records"):
            yield _apply_transformer(self._transformer, row)

    async def length(self) -> int:
        """Return the count of currently visible rows."""
        if self._read_df is None:
            if not await self._provider.has(self._table_name):
                return 0
            self._read_df = await self._provider.read_dataframe(self._table_name)
        return len(self._read_df)

    async def has(self, row_id: str) -> bool:
        """Check whether this table contains the row ID."""
        async for row in self:
            if isinstance(row, dict) and str(row.get("id")) == str(row_id):
                return True
            if str(getattr(row, "id", "")) == str(row_id):
                return True
        return False

    async def write(self, row: dict[str, Any]) -> None:
        """Buffer one row until the table is closed."""
        self._write_rows.append(row)

    async def close(self) -> None:
        """Flush buffered rows through the provider."""
        if not self._write_rows:
            return
        new_df = pd.DataFrame(self._write_rows)
        if not self._truncate and await self._provider.has(self._table_name):
            new_df = pd.concat(
                [await self._provider.read_dataframe(self._table_name), new_df],
                ignore_index=True,
            )
        await self._provider.write_dataframe(self._table_name, new_df)
        self._write_rows = []
        self._read_df = None