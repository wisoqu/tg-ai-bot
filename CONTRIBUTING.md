# Contributing

Thanks for helping improve the bot starter kit.

## Before opening a pull request

1. Open an issue for a large change or design proposal.
2. Keep secrets, SQLite files, test credentials, and real chat data out of commits.
3. Add or update unit tests for behavior changes.
4. Run:

   ```sh
   python -m unittest discover -s tests -v
   python -m compileall -q .
   git diff --check
   ```

5. Document new environment variables and operator-facing behavior in both README files and `.env.example`.

Use clear, small changes where possible. Include Python version, OS, and deployment method in bug reports. Replace all tokens and private user content with placeholders before sharing logs.
