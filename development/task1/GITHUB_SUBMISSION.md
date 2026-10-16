# Submit Task 1 through GitHub Actions

This avoids building Docker images on your Mac. The manually triggered workflow
builds only `task1-retrieval`, runs the German spot check, and uploads the image
and submission metadata to UniAgent 2026 for team `vera`. It does not launch
remote evaluation. TIRA may include Git source in private reproducibility
metadata; the Docker build context remains `task1-retrieval`.

1. Regenerate your TIRA token if it has appeared in shared logs.
2. On GitHub, open **Settings → Secrets and variables → Actions → New repository
   secret**. Name it `TIRA_TOKEN` and paste your TIRA token as the value.
3. Open **Actions → Submit Task 1 to TIRA → Run workflow**. Select `main`.
4. Wait for **Build, spot-check, and upload Task 1** to finish. Its final output
   gives the generated software name in TIRA. A failed job does not confirm a
   completed submission; inspect its logs before retrying.
5. In TIRA, open UniAgent 2026 for team `vera`, select that software, choose the
   appropriate Task 1 dataset and resources, and click **RUN**.

The workflow never runs on pushes or pull requests. Each successful manual run
creates an actual submission. Keep tokens out of source files, command output,
and screenshots. GitHub stores the token as a secret; it is not passed to the
submission container. No GitHub packages or Docker Hub credentials are needed.

The CLI wrapper gives Docker requests a 600-second inactivity timeout instead
of the SDK default of 60 seconds, allowing large registry pushes to wait for
progress updates. This does not change the retrieval system or its case budget.
If a push still fails, inspect the error before starting a new workflow run.
