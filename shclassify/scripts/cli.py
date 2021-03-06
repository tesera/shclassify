import os
import click
import logging

from shclassify import Tree, log


usage_log_path = os.path.abspath(__file__) + '.log'
usage = logging.FileHandler(usage_log_path)
usage.setLevel(logging.INFO)
usage_fmt = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
usage.setFormatter(usage_fmt)
usage_log = logging.getLogger('usage')
usage_log.propagate = False
usage_log.addHandler(usage)

console = logging.StreamHandler()
console_fmt = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
console.setFormatter(console_fmt)
log.addHandler(console)


def create_output_path(ctx, param, value):
    path = value
    if path is None:
        path = ctx.params.get('observations_file')
        path += '.pred'
    return path

@click.command('shclassify',
               help=('Predict landcover class for \'OBSERVATIONS_FILE\''
                     ' using SLS HRIS model'))
@click.argument('observations-file', type=click.Path(exists=True))
@click.option('--delim', '-d', default=',',
              type=click.Choice([',', r'\t', ';']),
              help='field delimeter')
@click.option('--index-col', '-i', default=0, type=int,
              help='index of column with observation IDs - 0 is first column')
@click.option('--chunksize', '-c', default=100000, type=int,
              help='lines to read and predict at a time')
@click.option('--verbose', '-v', is_flag=True)
@click.option('--outfile', '-o', callback=create_output_path,
              type=click.Path(),
              help='path to use for output (prediction) data')
def cli(observations_file, delim, index_col, chunksize, verbose, outfile):
    msg = '%s invoked cli' %os.environ.get('USER', 'anonymous')
    usage_log.info(msg)

    level = logging.INFO if verbose else logging.WARNING
    console.setLevel(level)

    click.echo('Creating classification tree')
    tree = Tree()

    click.echo(
        'Predicting classes for observations in {}'.format(observations_file)
    )
    tree.predict_file(observations_file, outfile,
                      overwrite=False, index_col=index_col, sep=delim,
                      chunksize=chunksize)

    click.echo('Predictions saved to file: {}'.format(outfile))
