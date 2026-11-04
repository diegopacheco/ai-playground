from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import psycopg


@asynccontextmanager
async def tenant_tx(database_url: str, tenant: str, read_all: bool = False) -> AsyncGenerator[psycopg.AsyncCursor]:
    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute("SELECT set_config('app.tenant', %s, true)", (tenant,))
                await cur.execute("SELECT set_config('app.read_all', %s, true)", ("on" if read_all else "off",))
                yield cur
