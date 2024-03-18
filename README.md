# Dottxt internal GSM8K evaluator

*Note: this is an internal tool primarily used to quickly run experiments, the code reflects this*

## Basic usage

Right now you should start by just pip installing the package locally (I uses `poetry` but I don't think that's strictly necessary). So in the project root directly either:

`pip install -e .`
 
 or

 `poetry run pip install -e .`

 Clearly you don't need to run in edit mode if you're not planning on developing locally, but for internal use that's rarely not the case.


 ## Running

 Right now everything is focused on just using the `gsm8k` data set, so there is no way to change the data set.

 Many other parameter are able to be configured by the command line script `./scripts/run_eval.py`. You can get a full list of options by viewing the source code or running:

 `python ./scripts/run_eval.py --help`

 *Note:* I do most of my work on a Mac Studio, so the default device is `mps`, this should be changed to `cuda` for most other users.

While there are many configuration options only a few are really needed to reproduce the results from [Structured Generation Improves LLM Performance](https://blog.dottxt.co/performance-gsm8k.html)

First run the unstructured evaluation...

```
python ./scripts/run_eval.py -i 0 -n 1319 --model <MODEL_NAME> --prompt standard_8 --struct unstruct_qa --db experiments.db
```

The run the structured evaluation...

```
python ./scripts/run_eval.py -i 0 -n 1319 --model <MODEL_NAME>  --prompt standard_8 --struct regex_qa_50_700 --db experiments.db
```

More details can be found by browsing the source code or using `--help`.

### Output

There is *a lot* of debugging output that is printed that cannot be toggled currently. It's not very clean, and the output while running the evals is a bit wonky as I've added a bunch of features for things like batching and using multiple samples that broke the original output. 

You can largely ignore it. At the end of a run you'll get a print out of the results of a single evaulation.

### Results

At the end of each run you should get an output that looks like this:

```
mistralai/Mistral-7B-v0.1-standard-regex_qa_300 5 obs - MAJ_ACC: 0.8000 PASS_ACC: 0.8000 
```

Which follows this structure:

```
{model} {number of observatiosn} obs - MAJ_ACC: {majority vote accuracy} PASS_ACC {any pass accuracy}
```

For replicating the results found in the article just we're using 1 samples so maj@1 is just the same as accuracy.

Additionally many of the key results are automatically stored in a sql data base set by the `--db` flag (results.db by default).

The `./scripts/leaderboard.py` script will print out a leaderboard of the models run. Here is an example:

```python
python ./scripts/leaderboard.py --db demo
model_name               |sub_set|prompt  |struct         |sampler|num_samples|total|maj_acc|pass_acc
-----------------------------------------------------------------------------------------------------
mistralai/Mistral-7B-v0.1|test   |standard|regex_qa_300   |greedy |1          |5    |0.8    |0.8     
mistralai/Mistral-7B-v0.1|test   |standard|regex_qa_50_700|greedy |1          |5    |0.8    |0.8 
```

More details are stored in the db if you want to dive in!