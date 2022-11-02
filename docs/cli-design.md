# CLI Design

Options that are different per run:

1. Config file
2. Device batch size (depends on image size and local GPUs)
3. Whether to debug or not (1 process or default number of processes)
4. Master port (with a default of 12345)
5. Data path
6. Number of processes (needs to match number of GPUs)

Options that are different per machine:

1. Output directory
2. Virtual environment location
