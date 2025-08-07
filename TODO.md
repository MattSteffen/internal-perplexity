# TODO

## General

- Test the tools

## Deployment

- [ ] Set up helm values and docker stack charts for deployment
  - [ ] It should be separate deployments, but have a script for each like `./deploy.sh open-webui` and `./deploy.sh ollama`
  - [ ] test
- [ ] Add user valves where they can put in username and password for database token

## Other

- use cloudflared tunnel --url http://localhost:<desired-port> to expose the service to the internet
- maybe can do ingresses exposing http://ollama.localhost:5000 and similarly for the other services.

https://github.com/PatrickJS/awesome-cursorrules/blob/main/rules/htmx-go-basic-cursorrules-prompt-file/.cursorrules
