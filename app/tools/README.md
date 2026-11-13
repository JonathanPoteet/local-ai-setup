# Tools

The current app routes requests to purpose-built tools using rules and embedding
similarity, or starts them from the UI. A goal for this project is to evolve
toward a model-driven agent that can choose among registered tools based on the
conversation. Tool access will still be limited and validated by the
application.

| Tool | What it does | Network |
| --- | --- | --- |
| Web search | Looks up current or external information using DuckDuckGo, Bing, and Google News. | Internet required |
| Headlines | Returns recent headlines for broad news requests. | Internet required |
| Reference lookup | Returns search-result excerpts for direct “tell me about…” requests. | Internet required |
| Page fetch | Extracts text from a public HTTP(S) page submitted in the UI. | Internet required |
| Knowledge retrieval | Finds relevant passages in indexed local documents and saved pages. | Local |

## Exploratory TODOs

These are possible next steps, not a committed roadmap:

- [ ] Define a small, consistent tool interface for matching requests, running
  tools, and returning answers, sources, and errors.
- [ ] Add focused tests for tool routing, successful results, disabled tools,
  and provider failures.
- [ ] Explore model-driven tool calling with a local model that supports it;
  keep tool permissions and input validation in application code.
- [ ] Choose a concrete first machine-learning tool use case before adding
  another model or dependency. Possible directions include text
  classification, extraction, or reranking.
- [ ] Make tool selection and availability easier to inspect and tune without
  changing normal chat behavior.
- [ ] Document which tools need network access, local models, or indexed data.
