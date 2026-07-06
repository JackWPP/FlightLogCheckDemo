# Online Debugging And Logs

This demo writes structured JSONL logs for request, task, pipeline stage, and provider call debugging.

## Log Location

By default logs are written to:

```text
outputs/runtime/logs/app.jsonl
```

Useful environment variables:

```text
APP_LOG_LEVEL=INFO
APP_LOG_STDOUT=1
APP_LOG_DIR=
APP_LOG_FILE=app.jsonl
APP_LOG_MAX_BYTES=10485760
APP_LOG_BACKUP_COUNT=7
TASK_LEASE_SECONDS=1800
```

API keys and authorization-like values are redacted by the logger.

## How To Debug One Online Task

1. Get the `task_id` from the UI task card or `/api/tasks?session_id=...`.
2. Fetch task state:

```bash
curl http://127.0.0.1:8000/api/tasks/<task_id>
```

3. Fetch related logs:

```bash
curl http://127.0.0.1:8000/api/tasks/<task_id>/logs
```

4. Open generated evidence under:

```text
out/<task_id>/
```

Important files:

```text
report.json
field_candidates.json
ppocrv6/job.json
ppocrv6/ocr_blocks.json
ppocrv6/error.json
ppocr_rois/
roi_reviews/
```

## Events To Look For

Pipeline:

```text
pipeline.start
pipeline.ocr.running
pipeline.ocr.done / pipeline.ocr.failed
pipeline.validate.done
pipeline.review.running / pipeline.review.done
pipeline.issue_triage.done
pipeline.done
```

PaddleOCR:

```text
ppocr.cache_hit
ppocr.submit.start / ppocr.submit.done
ppocr.poll.start / ppocr.poll.done
ppocr.download.start
ppocr.done
ppocr.exception
```

Cleaner:

```text
cleaner.start
cleaner.section.start / cleaner.section.done
cleaner.section.cache_hit
cleaner.section.failed
cleaner.timeout
cleaner.done
```

ROI VLM:

```text
vlm.request.start
vlm.request.done
vlm.request.failed
vlm.skipped
```

Task worker:

```text
task.created
task.claimed
task.progress
task.finished
task.failed
task.recovered_stale
```

## Common Diagnosis

- Slow OCR: check `ppocr.poll.done.duration_ms` and `report.timings.ppocr_poll_ms`.
- Cleaner timeout: check `cleaner.section.failed`, `cleaner.timeout`, and `cleaner.done.fallback_sections`.
- ROI VLM timeout: check `vlm.request.failed` with `field_id`.
- Bad field evidence: inspect `out/<task_id>/ppocr_rois/<field_id>.png` and `out/<task_id>/roi_reviews/<field_id>.png`.
- Stuck task after restart: startup should emit `task.recovered_stale` and return expired running tasks to pending.

## Architecture Note

The online path should prefer `/api/tasks` and `/api/tasks/batch`. The older synchronous `/api/analyze` and `/api/analyze/stream` paths are kept for compatibility and manual experiments.
