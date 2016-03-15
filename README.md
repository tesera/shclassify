# SHClassify

## Background

shclassify is a python library and CLI to run the SLS HRIS Land Cover Classification model.

The general approach is as follows:

1. load a set of observations
2. create a data frame to hold predictions (using the same index as the observations)
2. subset the observations to those belonging to the class at the highest level of the heirarchy which has not been predicted yet
3. calculate the class membership logits using using the model belonging to this class
4. transform the logits to probabilities
5. select the class (using the multinomial or binary heuristic) 
6. update the data frame of predictions
7. repeat 2-6 until no observations have a predicted class for which there is another model to apply

## Installation

    unzip shclassify.zip
    # git clone
    cd shclassify
    pip install -e shclassify

## Usage

The CLI is the recommended usage

    > shclassify --help                                                          [git][shclassify/.][masterU]
    Usage: shclassify [OPTIONS] OBSERVATIONS_FILE

      Predict landcover class for 'OBSERVATIONS_FILE' using SLS HRIS model

    Options:
      -d, --delim [,|\t|;]     field delimeter
      -i, --index-col INTEGER  index of column with observation IDs - 0 is first
                               column
      -c, --chunksize INTEGER  lines to read and predict at a time
      -v, --verbose
      -o, --outfile PATH       path to use for output (prediction) data
      --help                   Show this message and exit.


## Development

To modify, create a feature branch, make the desired modification. Run the tests to check for regressions, and add tests to ensure new code works as expected and to protect the new code from regressions.

    py.test ./tests

See http://pytest.org/latest/ for more details about testing.

## Troubleshooting

If you are having issues with the CLI, try running with verbose mode `-v`. Issues often arise from errors in, or mismatches between, data and model files.

## Docs

Beyond the CLI, documentation can be found at docs/build/index.html. Advanced users may also refer to the source code.
