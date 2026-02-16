#!/usr/bin/env python3
"""Database utilities for StereoWipe.

This module provides command-line utilities for interacting with the StereoWipe database.
"""

import click
import json
from typing import Optional
from database import Database


@click.group()
def cli():
    """StereoWipe database utilities."""
    pass


@cli.command()
def summary():
    """Display summary statistics from the database."""
    db = Database()
    stats = db.get_evaluation_summary()
    
    click.echo("\n=== StereoWipe Database Summary ===")
    click.echo(f"Total evaluations: {stats['total_evaluations']}")
    
    if stats['evaluations_by_model']:
        click.echo("\nEvaluations by model:")
        for model, count in stats['evaluations_by_model'].items():
            click.echo(f"  - {model}: {count} evaluations")
        
        click.echo("\nStereotype statistics by model:")
        for model, model_stats in stats['stereotype_stats'].items():
            click.echo(f"  - {model}:")
            click.echo(f"    * Stereotype rate: {model_stats['stereotype_rate']:.1f}%")
            if model_stats['avg_severity'] is not None:
                click.echo(f"    * Average severity: {model_stats['avg_severity']:.2f}")
    else:
        click.echo("No evaluations found in database.")


@cli.command()
@click.option('--model', help='Filter by model name')
@click.option('--limit', default=10, help='Number of evaluations to show')
def list_evaluations(model: Optional[str], limit: int):
    """List recent evaluations from the database."""
    db = Database()
    
    if model:
        evaluations = db.get_evaluations_by_model(model)[:limit]
        click.echo(f"\n=== Recent evaluations for {model} ===")
    else:
        evaluations = db.get_all_evaluations(limit=limit)
        click.echo("\n=== Recent evaluations (all models) ===")
        
    for eval in evaluations:
        click.echo(f"\nID: {eval['id']}")
        click.echo(f"Prompt: {eval['prompt_text'][:50]}...")
        click.echo(f"Response: {eval['response_text'][:50]}...")
        click.echo(f"Is stereotype: {bool(eval['is_stereotype'])}")
        if eval['severity_score']:
            click.echo(f"Severity: {eval['severity_score']}")
        click.echo(f"Judge: {eval['judge_name']}")
        click.echo(f"Date: {eval['created_at']}")


@cli.command()
@click.argument('table', type=click.Choice(['evaluations', 'human_annotations', 'arena_battles']))
@click.argument('output_file')
@click.option('--format', 'output_format', type=click.Choice(['csv', 'json']), default='json')
def export(table: str, output_file: str, output_format: str):
    """Export data from a database table."""
    db = Database()
    
    if output_format == 'csv':
        db.export_to_csv(table, output_file)
    else:
        db.export_to_json(table, output_file)
    
    click.echo(f"Exported {table} to {output_file}")


@cli.command()
def list_unannotated():
    """List responses that need human annotation."""
    db = Database()
    unannotated = db.get_unannotated_responses(limit=20)
    
    click.echo(f"\n=== Unannotated responses ({len(unannotated)} found) ===")
    for item in unannotated:
        click.echo(f"\nPrompt ID: {item['prompt_id']}")
        click.echo(f"Prompt: {item['prompt_text'][:80]}...")
        click.echo(f"Response: {item['response_text'][:80]}...")


@cli.command()
def arena_stats():
    """Display arena battle statistics."""
    db = Database()
    stats = db.get_model_win_stats()
    
    if not stats:
        click.echo("No arena battles found in database.")
        return
    
    click.echo("\n=== Arena Battle Statistics ===")
    for model, model_stats in sorted(stats.items(), key=lambda x: x[1]['wins'], reverse=True):
        total = model_stats['total']
        if total > 0:
            win_rate = (model_stats['wins'] / total) * 100
            click.echo(f"\n{model}:")
            click.echo(f"  Wins: {model_stats['wins']} ({win_rate:.1f}%)")
            click.echo(f"  Losses: {model_stats['losses']}")
            click.echo(f"  Ties: {model_stats['ties']}")
            click.echo(f"  Total battles: {total}")


if __name__ == '__main__':
    cli()