## Now

## Deployment

- [ ] Set up helm values and docker stack charts for deployment
  - [ ] It should be separate deployments, but have a script for each like `./deploy.sh open-webui` and `./deploy.sh ollama`
  - [ ] test
- [ ] Add user valves where they can put in username and password for database token

- Crawler
- [x] Implement a good converter.
  - [x] Determine how to ingest PDFs and other documents. Should be able to export to markdown.
- [ ] logging

- Radchat
  - [ ] Tool calling needs to work, it didn't work with qwen3:1.7b but works as intended with qwen3:latest (8b)
  - [ ] Prompt needs to better teach how to use the metadata filters, this is consistently wrong.
  - [ ] Make sure that after a tool call, the original user's prompt is front and center, that way the retrieved documents don't completely override the conversation.
  - [ ] Migrate radchat to milvus_search so it is cleaner.
    - [ ] Also create an XMChat that does the fancy 4 partition search.

## Other

- use cloudflared tunnel --url http://localhost:<desired-port> to expose the service to the internet
- maybe can do ingresses exposing http://ollama.localhost:5000 and similarly for the other services.

https://github.com/PatrickJS/awesome-cursorrules/blob/main/rules/htmx-go-basic-cursorrules-prompt-file/.cursorrules
