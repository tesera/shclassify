import click
import logging

import shclassify


console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger(__name__).addHandler(console)

log = logging.getLogger(__name__)


@click.command('shclassify')
@click.argument('count', type=int, metavar='N')
def cli(count):
    """Echo a value `N` number of times"""
    for i in range(count):
        log.info('logged info')
        click.echo(shclassify.has_legs())
