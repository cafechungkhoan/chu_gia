### Lê Đặng Trung Hiếu
#### Supervisor | ABS, Viet Capital Securities Joint Stock Company (VCSC)
- :zap: I love math, stock market, marketing and data science.
- 🌱 I’m addicted to learning and growing every day
- :earth_africa: I am currently sharing a little bit of my knowledge to the world through my website
- 📫 How to find me: 
  - :bulb: [Website articles](http://cafechungkhoan.com/)
  - :office: [LinkedIn](https://www.linkedin.com/in/hi%E1%BA%BFu-l%C3%AA-%C4%91%E1%BA%B7ng-trung-b968014b/)
- 📫 My top skill: 
  - :atom: Search Engine Optimization (SEO)
  - :atom: Digital Ads Manager, Digital Strategies
- :atom: Python (Programming Language) for Data Science & Machine Learning


## Vendor task tracker

`vendor_task_tracker.py` is a small CLI to help you keep track of vendor deliverables and chase overdue work.

### Basic usage

```bash
# Add a new task for a vendor
python vendor_task_tracker.py add --vendor "ABC Studio" --description "Final TVC storyboard" --due 2024-07-01 --notes "Need draft before board meeting"

# List everything
python vendor_task_tracker.py list

# Check only overdue or soon-due work (default 3 days ahead)
python vendor_task_tracker.py remind --days 5

# Update a task when you receive an update from the vendor
python vendor_task_tracker.py update 1 --status in_progress --notes "Sent reminder on Monday"
```

All tasks are stored in `vendor_tasks.json` (you can point to a different file with `--data-file`). This makes it easy to version-control the follow-up log with Git or share it via cloud storage.

  
