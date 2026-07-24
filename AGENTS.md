## Secret handling

- Never read, print, search, copy, modify, or inspect `.env` files.
- Never run commands such as `cat .env`, `type .env`, `Get-Content .env`,
  `grep` against `.env`, or scripts that print environment variables.
- Use `.env.example` only to determine required variable names.
- Never display API keys, tokens, credentials, or secrets in logs or responses.
- Ask the user to run API-dependent tests manually when credentials are required.