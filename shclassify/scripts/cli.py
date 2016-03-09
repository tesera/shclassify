import os
import click
import logging


console = logging.StreamHandler()
console.setLevel(logging.INFO)
console_fmt = logging.Formatter('%(levelname)-8s %(message)s')
console.setFormatter(console_fmt)
console_log = logging.getLogger('console')
console_log.propagate = False
console_log.addHandler(console)

usage_log_path = os.path.abspath(__file__) + '.log'
usage = logging.FileHandler(usage_log_path)
usage.setLevel(logging.INFO)
usage_fmt = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
usage.setFormatter(usage_fmt)
usage_log = logging.getLogger('usage')
usage_log.propagate = False
usage_log.addHandler(usage)


def create_output_path(ctx, param, value):
    path = value
    if path is None:
        path = ctx.params.get('path')
        path += '.out'
    return path

@click.command('shclassify')
@click.argument('path', type=click.Path(exists=True), is_eager=True)
@click.option('--delim', '-d', default=',', type=click.Choice([',', r'\t', ';']),
              help='field delimeter')
@click.option('--verbose', '-v', is_flag=True)
@click.option('--intermediate-preds', '-i', is_flag=True,
              help='save intermediate class predictions')
@click.option('--outfile', '-o', callback=create_output_path, type=click.Path())
def cli(path, delim, intermediate_preds, verbose, outfile):
    msg = '%s invoked cli' %os.environ.get('USER', 'anonymous')
    usage_log.info(msg)

    console_log.info('path: %s' %path)
    console_log.info('outfile: %s' %outfile)
    console_log.info('delim: %s' %delim)
    console_log.info('verbose: %s' %verbose)
    console_log.info('save intermediate preds: %s' \
                     %intermediate_preds)
