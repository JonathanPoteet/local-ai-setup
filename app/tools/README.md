# Tools

The model chooses tools from a registry when they are available and useful.
The orchestrator validates tool calls, limits execution rounds, and collects
results through one shared result format. Web tools are omitted when web search
is disabled; creative requests do not receive tools.

| Tool | What it does | Network |
| --- | --- | --- |
| `web_search` | Looks up general, current, or explicitly requested information using DuckDuckGo, Bing, and Google News. | Internet required |
| `headlines` | Retrieves recent news results, optionally for a specific topic. | Internet required |
| `reference_lookup` | Finds concise encyclopedia summaries through Wikipedia's REST API. | Internet required |
| `search_knowledge` | Finds passages in indexed local documents and saved pages. | Local |
| Page fetch | Extracts text from a public HTTP(S) page submitted in the UI. | Internet required |

Tool results share a JSON contract: `status`, `results`, and `error`. Each result
has `title`, `url`, `snippet`, `source`, and `published` fields; fields that do
not apply are `null`. The UI's page-fetch action remains a direct user action,
not an agent tool.
Chat responses expose selected tool names in `tools_used`; web-tool usage is
also reflected by `web_search_performed`.
Local-place searches inspect up to five result pages using the same public-URL,
redirect, content-type, and response-size safeguards as page fetching. Sites
that block extraction are identified in the returned evidence.

## Exploratory TODOs

These are possible next steps, not a committed roadmap:

- [x] Define a consistent result contract for model-callable tools.
- [ ] Add focused tests for tool routing, successful results, disabled tools,
  and provider failures.
- [x] Verify model-driven tool calling with the local model.
- [x] Register web search, headlines, reference lookup, and knowledge search
  with application-controlled availability and bounded execution.
- [ ] Choose a concrete first machine-learning tool use case before adding
  another model or dependency. Possible directions include text
  classification, extraction, or reranking.
- [ ] Make tool selection and availability easier to inspect and tune without
  changing normal chat behavior.
- [ ] Document which tools need network access, local models, or indexed data.
