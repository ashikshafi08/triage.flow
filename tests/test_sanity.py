import asyncio, aiohttp, json, os
from src.issue_rag import IssueAwareRAG   # assumes your src path
CSV = "smolagents_eval_queries.csv"
OWNER, REPO = "huggingface", "smolagents"

async def main():
    rag = IssueAwareRAG(OWNER, REPO)
    await rag.initialize(force_rebuild=False)   # use existing index

    async with aiohttp.ClientSession() as s:
        async def check(row):
            num = int(row["expected_issue_number"])
            # 1) Does the issue page exist?
            r = await s.get(f"https://github.com/{OWNER}/{REPO}/issues/{num}")
            exists = r.status == 200
            # 2) Did we index it?
            in_index = num in rag.indexer.issue_docs
            return num, exists, in_index

        import csv, asyncio
        with open(CSV) as f: rows = list(csv.DictReader(f))
        results = await asyncio.gather(*(check(r) for r in rows))

    print("\n# Eval sanity report")
    for num, ok_page, in_idx in results:
        print(f"{num:<7} page:{'✅' if ok_page else '❌'}   indexed:{'✅' if in_idx else '❌'}")

asyncio.run(main())
