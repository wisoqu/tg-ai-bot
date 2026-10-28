# Security policy

## Secrets and user data

Never publish Telegram tokens, Gemini API keys, `.env` files, database files, Telegram payment payloads, or private chat content in an issue or log. Revoke exposed provider keys immediately. Report a security vulnerability privately to the repository owner before public disclosure.

The bot stores received chat text and explicit saved facts in its local SQLite database. Operators must secure the host and backups, tell group members when message context is enabled, honor applicable consent/privacy requirements, and provide a contact channel for deletion/support. API requests include selected conversation context; addressed group voice notes are sent to Gemini for transcription.

## Deployment checklist

- Keep `.env` owner-readable and outside version control.
- Use a dedicated Google Cloud project/key, inspect its AI Studio quotas, and configure provider billing caps/alerts if billing is enabled.
- Keep the bot token and database out of container images and public backups.
- Limit server and database access; update dependencies and the base image regularly.
- Set conservative app-wide quotas. The app cannot guarantee provider costs or availability, especially if operators explicitly configure billable models.
- Test Stars purchases with a test bot before enabling packages for real users.
