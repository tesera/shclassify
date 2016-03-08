import click

import shclassify


@click.command('shclassify')
@click.argument('count', type=int, metavar='N')
def cli(count):
    """Echo a value `N` number of times"""
    for i in range(count):
        click.echo(shclassify.has_legs)
