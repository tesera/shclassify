import os
import click
import logging

import shclassify


console = logging.StreamHandler()
console.setLevel(logging.INFO)
console_fmt = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(console_fmt)
console_log = logging.getLogger('console')
console_log.propagate = False
console_log.addHandler(console)

usage_log_path = os.path.abspath(__file__) + '.log'
usage = logging.FileHandler(usage_log_path)
usage.setLevel(logging.INFO)
usage_fmt = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
usage.setFormatter(usage_fmt)
usage_log = logging.getLogger('usage')
usage_log.propagate = False
usage_log.addHandler(usage)

@click.command('shclassify')
@click.argument('count', type=int, metavar='N')
def cli(count):
    """Echo a value `N` number of times"""
    usage_log.info('cli invoked')
    for i in range(count):
        console_log.info('logged info')
        click.echo(shclassify.has_legs())
