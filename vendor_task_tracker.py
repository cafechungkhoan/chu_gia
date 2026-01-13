"""Command-line tool to track vendor tasks.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

DEFAULT_DATA_FILE = Path(__file__).with_name("vendor_tasks.json")
VALID_STATUSES = {"pending", "in_progress", "blocked", "done"}


@dataclass
class VendorTask:
    """Represents a task assigned to an external vendor."""

    id: int
    vendor: str
    description: str
    due_date: str
    status: str = "pending"
    notes: str | None = None

    @property
    def due(self) -> date:
        return date.fromisoformat(self.due_date)


class TaskStorage:
    """Utility class to load and persist tasks in JSON format."""

    def __init__(self, file_path: Path = DEFAULT_DATA_FILE):
        self.file_path = file_path

    def load(self) -> List[VendorTask]:
        if not self.file_path.exists():
            return []
        with self.file_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return [VendorTask(**item) for item in raw]

    def save(self, tasks: List[VendorTask]) -> None:
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(task) for task in tasks], f, indent=2, ensure_ascii=False)


def parse_due_date(raw_date: str) -> str:
    try:
        return date.fromisoformat(raw_date).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Due dates must follow the ISO format YYYY-MM-DD"
        ) from exc


def ensure_status(status: str) -> str:
    normalized = status.lower()
    if normalized not in VALID_STATUSES:
        raise argparse.ArgumentTypeError(
            f"Status must be one of {', '.join(sorted(VALID_STATUSES))}"
        )
    return normalized


def add_task(args: argparse.Namespace, storage: TaskStorage) -> None:
    tasks = storage.load()
    next_id = max([task.id for task in tasks], default=0) + 1
    due_date = parse_due_date(args.due)
    task = VendorTask(
        id=next_id,
        vendor=args.vendor,
        description=args.description,
        due_date=due_date,
        status=args.status,
        notes=args.notes,
    )
    tasks.append(task)
    storage.save(tasks)
    print(f"Added task #{task.id} for vendor '{task.vendor}'.")


def list_tasks(args: argparse.Namespace, storage: TaskStorage) -> None:
    tasks = storage.load()
    if args.vendor:
        tasks = [task for task in tasks if task.vendor.lower() == args.vendor.lower()]
    if args.status:
        tasks = [task for task in tasks if task.status == args.status]
    tasks.sort(key=lambda t: (t.due, t.vendor))

    if not tasks:
        print("No tasks found.")
        return

    headers = ("ID", "Vendor", "Due", "Status", "Description", "Notes")
    column_widths = [6, 20, 12, 12, 40, 30]

    def format_row(values: List[str]) -> str:
        padded = [str(value)[:width].ljust(width) for value, width in zip(values, column_widths)]
        return " | ".join(padded)

    print(format_row(headers))
    print("-" * (sum(column_widths) + 3 * (len(headers) - 1)))
    for task in tasks:
        print(
            format_row(
                [
                    task.id,
                    task.vendor,
                    task.due_date,
                    task.status,
                    task.description,
                    task.notes or "",
                ]
            )
        )


def update_task(args: argparse.Namespace, storage: TaskStorage) -> None:
    tasks = storage.load()
    task = next((task for task in tasks if task.id == args.id), None)
    if task is None:
        raise SystemExit(f"Task with ID {args.id} not found.")

    if args.vendor:
        task.vendor = args.vendor
    if args.description:
        task.description = args.description
    if args.due:
        task.due_date = parse_due_date(args.due)
    if args.status:
        task.status = args.status
    if args.notes is not None:
        task.notes = args.notes

    storage.save(tasks)
    print(f"Updated task #{task.id}.")


def remind_tasks(args: argparse.Namespace, storage: TaskStorage) -> None:
    tasks = storage.load()
    today = date.today()
    window = today + timedelta(days=args.days)

    pending_tasks = [task for task in tasks if task.status != "done" and task.due <= window]
    pending_tasks.sort(key=lambda t: t.due)

    if not pending_tasks:
        print(f"No tasks due within the next {args.days} day(s).")
        return

    for task in pending_tasks:
        delta = (task.due - today).days
        status = "due today" if delta == 0 else ("overdue" if delta < 0 else f"due in {delta} day(s)")
        print(
            f"#{task.id} | {task.vendor} | {task.due_date} | {status} | {task.description}"
        )
        if task.notes:
            print(f"    notes: {task.notes}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track vendor deliverables and reminders.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Optional path to a custom JSON file for storing tasks.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a new vendor task")
    add_parser.add_argument("--vendor", required=True)
    add_parser.add_argument("--description", required=True)
    add_parser.add_argument("--due", required=True, help="Due date in YYYY-MM-DD format")
    add_parser.add_argument(
        "--status",
        default="pending",
        type=ensure_status,
        help="Initial status (pending, in_progress, blocked, done)",
    )
    add_parser.add_argument("--notes", help="Optional notes for chasing the vendor")
    add_parser.set_defaults(func=add_task)

    list_parser = subparsers.add_parser("list", help="List vendor tasks")
    list_parser.add_argument("--vendor", help="Filter tasks by vendor name")
    list_parser.add_argument(
        "--status",
        choices=sorted(VALID_STATUSES),
        help="Filter tasks by status",
    )
    list_parser.set_defaults(func=list_tasks)

    update_parser = subparsers.add_parser("update", help="Update an existing task")
    update_parser.add_argument("id", type=int, help="Task ID")
    update_parser.add_argument("--vendor")
    update_parser.add_argument("--description")
    update_parser.add_argument("--due")
    update_parser.add_argument(
        "--status",
        type=ensure_status,
        help="New status value",
    )
    update_parser.add_argument(
        "--notes",
        help="Replace the notes. Provide an empty string to clear existing notes.",
    )
    update_parser.set_defaults(func=update_task)

    remind_parser = subparsers.add_parser(
        "remind", help="Show tasks that are due soon so you can chase vendors"
    )
    remind_parser.add_argument(
        "--days", type=int, default=3, help="Number of days to look ahead for due tasks"
    )
    remind_parser.set_defaults(func=remind_tasks)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    storage = TaskStorage(args.data_file)
    args.func(args, storage)


if __name__ == "__main__":
    main()
