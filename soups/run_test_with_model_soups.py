"""Run test with model soups"""

import argparse
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torchvision
from torch.utils.data import DataLoader

import soups.utils as utils
from soups.opts import add_test_with_model_soups_opts
from soups.utils.logger import init_logger, logger
from soups.utils.training import (
    EvalResults,
    convert_eval_results_to_dict,
    eval_model,
    make_model,
    print_eval_results,
)


@dataclass
class Candidate:
    model_path: str
    val_results: EvalResults


EPS = 1e-7


def test_with_model_soups(args: argparse.Namespace) -> None:
    if os.path.isdir(args.output_dir):
        logger.error(f'Output directory already exists: {args.output_dir}')
        exit(1)

    os.makedirs(args.output_dir)
    log_file_path = os.path.join(args.output_dir, 'test_with_model_soups.log')
    init_logger(log_file=log_file_path, compact=True)

    utils.set_seed(args.seed)
    logger.info(f'Using seed: {args.seed}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # find all model checkpoint files
    model_paths = utils.find_checkpoint_files(checkpoint_files_or_dirs=args.checkpoint_paths)
    if not model_paths:
        logger.error('No model checkpoints found.')
        exit(1)

    # remove duplicate checkpoints from a single epoch
    if args.remove_duplicate_checkpoints:
        logger.info('Removing duplicate checkpoints...')
        seen_epochs: set[int] = set()
        filtered_model_paths: list[str] = []
        for model_path in model_paths:
            model_path_basename = os.path.basename(model_path)
            if not model_path_basename.startswith('model_epoch_'):
                logger.error(
                    'Expected model checkpoints to be named as '
                    'model_epoch_{epoch}_{metric}_{metric_value:.4f}.pth. '
                    f'Found {model_path_basename}'
                )
                exit(1)
            try:
                epoch_number = int(model_path_basename[len('model_epoch_') :].split('_')[0])
                if epoch_number in seen_epochs:
                    logger.warning(
                        f'Duplicate checkpoint for epoch {epoch_number}, ignoring {model_path}'
                    )
                else:
                    seen_epochs.add(epoch_number)
                    filtered_model_paths.append(model_path)
            except Exception:
                logger.error(f'Failed to extract epoch number from checkpoint: {model_path}')
                exit(1)
        model_paths = filtered_model_paths

    num_models = len(model_paths)
    logger.info(f'Found total {len(model_paths)} unique checkpoints for cooking')

    # test dataset and test data loader
    eval_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, 'val'),
        transform=eval_transforms,
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, 'test'),
        transform=eval_transforms,
    )
    class_names = test_dataset.classes
    num_classes = len(class_names)

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(1, min(16, (os.cpu_count() or 1) // 2)),
        pin_memory=True,
        persistent_workers=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(1, min(16, (os.cpu_count() or 1) // 2)),
        pin_memory=True,
        persistent_workers=True,
    )

    candidates: list[Candidate] = []
    model = make_model(
        model_name=args.model,
        num_classes=num_classes,
    ).to(device)
    for i, model_path in enumerate(model_paths):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f'Evaluating model [{i + 1}/{num_models}]: {model_path}')
        val_results = eval_model(
            model=model,
            eval_data_loader=val_data_loader,
            device=device,
            num_classes=num_classes,
        )
        candidates.append(Candidate(model_path, val_results))
        print_eval_results(val_results, prefix='val')

    assert len(candidates) == len(model_paths)
    # uniform soup (i.e. mixing all models together)
    if args.uniform_soup:
        logger.info('** Start cooking uniform soup **')
        uniform_soup_result_file = os.path.join(
            args.output_dir,
            'uniform_soup_results.json',
        )

        result_data = {}
        result_data['models'] = {}
        result_data['num_models'] = len(candidates)
        for candidate in candidates:
            result_data['models'][candidate.model_path] = convert_eval_results_to_dict(
                eval_results=candidate.val_results, class_names=class_names
            )

        uniform_soup_params = {}
        for i, model_path in enumerate(model_paths):
            logger.info(f'[{i + 1}/{num_models}] Adding model {model_path} to uniform soup')
            model_state_dict = torch.load(model_path, map_location=device)['model_state_dict']
            if i == 0:
                uniform_soup_params = {
                    k: v.clone() * (1.0 / num_models) for k, v in model_state_dict.items()
                }
            else:
                uniform_soup_params = {
                    k: v.clone() * (1.0 / num_models) + uniform_soup_params[k]
                    for k, v in model_state_dict.items()
                }

        # test the uniform soup
        model.load_state_dict(uniform_soup_params)
        test_results = eval_model(
            model=model,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        print('** Uniform soup test results: **')
        print_eval_results(eval_results=test_results, prefix='test')

        # save the results
        result_data['uniform_soup'] = convert_eval_results_to_dict(
            eval_results=test_results, class_names=class_names
        )
        with open(uniform_soup_result_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        logger.info(f'Uniform soup results saved to {uniform_soup_result_file}')

        # save the soup model
        uniform_soup_model_path = os.path.join(args.output_dir, 'uniform_soup.pth')
        torch.save(
            {
                'model_state_dict': uniform_soup_params,
                'test_results': test_results,
            },
            uniform_soup_model_path,
        )

    if args.greedy_soup:
        comp_metric = args.greedy_soup_comparison_metric
        assert comp_metric in ['accuracy', 'precision', 'recall', 'f1', 'loss'], (
            f'Invalid comparison metric: {comp_metric}'
        )
        logger.info('** Start cooking greedy soup **')
        logger.info(f'Comparison metric: {comp_metric}')
        greedy_soup_result_file = os.path.join(
            args.output_dir,
            'greedy_soup_results.json',
        )

        current_candidates = deepcopy(candidates)
        # for loss, we want to minimize it
        if comp_metric == 'loss':
            for candidate in current_candidates:
                candidate.val_results['loss'] = -candidate.val_results['loss']

        # sort models by decreasing `comp_metric`
        current_candidates = sorted(
            current_candidates, key=lambda item: item.val_results[comp_metric], reverse=True
        )

        result_data = {}
        result_data['models'] = {}
        result_data['num_models'] = len(current_candidates)
        for candidate in current_candidates:
            result_data['models'][candidate.model_path] = convert_eval_results_to_dict(
                eval_results=candidate.val_results, class_names=class_names
            )

        # start the soup by using the first ingredient.
        greedy_soup_ingredients = [current_candidates[0].model_path]
        greedy_soup_params = torch.load(
            current_candidates[0].model_path,
            map_location=device,
        )['model_state_dict']
        best_val_result_so_far = current_candidates[0].val_results[comp_metric]

        for i in range(1, num_models):
            logger.info(f'Trying model [{i + 1}/{num_models}] {current_candidates[i].model_path}')

            # get the potential new soup by adding the current model
            new_ingredient_params = torch.load(
                current_candidates[i].model_path,
                map_location=device,
            )['model_state_dict']
            num_ingredients = len(greedy_soup_ingredients)
            potential_greedy_soup_params = add_ingredient_to_soup(
                soup=greedy_soup_params,
                num_ingredients=num_ingredients,
                ingredient=new_ingredient_params,
            )

            # test the new-branch model
            model.load_state_dict(potential_greedy_soup_params)
            cur_val_result = eval_model(
                model=model,
                eval_data_loader=val_data_loader,
                device=device,
                num_classes=num_classes,
            )[comp_metric]
            if comp_metric == 'loss':
                cur_val_result = -cur_val_result

            # if `comp_metric` improves, add the model to the soup
            logger.info(
                f'Potential greedy soup val {comp_metric} {abs(cur_val_result):0.6f}, '
                f'best so far {abs(best_val_result_so_far):0.6f}.'
            )
            if cur_val_result + EPS > best_val_result_so_far:
                greedy_soup_ingredients.append(current_candidates[i].model_path)
                best_val_result_so_far = cur_val_result
                greedy_soup_params = potential_greedy_soup_params
                logger.info(f'Added model {current_candidates[i].model_path} to greedy soup')

        result_data['ingredients'] = greedy_soup_ingredients
        result_data['num_ingredients'] = len(greedy_soup_ingredients)
        result_data[f'best_val_{comp_metric}'] = abs(best_val_result_so_far)

        # test the final greedy soup
        print('** Greedy soup ingredients: **')
        for ingredient in greedy_soup_ingredients:
            print(f'  - {ingredient}')

        model.load_state_dict(greedy_soup_params)
        test_results = eval_model(
            model=model,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        print('** Greedy soup test results: **')
        print_eval_results(eval_results=test_results, prefix='test')

        # save the results
        result_data['greedy_soup'] = convert_eval_results_to_dict(
            eval_results=test_results, class_names=class_names
        )
        with open(greedy_soup_result_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        logger.info(f'Greedy soup results saved to {greedy_soup_result_file}')

        # save the soup model
        greedy_soup_model_path = os.path.join(args.output_dir, 'greedy_soup.pth')
        torch.save(
            {
                'model_state_dict': greedy_soup_params,
                'test_results': test_results,
            },
            greedy_soup_model_path,
        )

    if args.pruned_soup:
        comp_metric = args.greedy_soup_comparison_metric
        assert comp_metric in ['accuracy', 'precision', 'recall', 'f1', 'loss'], (
            f'Invalid comparison metric: {comp_metric}'
        )
        logger.info('** Start cooking pruned soup **')
        logger.info(f'Comparison metric: {comp_metric}')

        pruned_soup_result_file = os.path.join(
            args.output_dir,
            'pruned_soup_results.json',
        )

        current_candidates = deepcopy(candidates)

        # for loss, we want to minimize it
        if comp_metric == 'loss':
            for candidate in current_candidates:
                candidate.val_results['loss'] = -candidate.val_results['loss']

        # sort models by decreasing `comp_metric`
        current_candidates = sorted(
            current_candidates, key=lambda item: item.val_results[comp_metric], reverse=True
        )

        # compute starting soup (uniform soup)
        result_data = {}
        result_data['models'] = {}
        result_data['num_models'] = len(current_candidates)
        for candidate in current_candidates:
            result_data['models'][candidate.model_path] = convert_eval_results_to_dict(
                eval_results=candidate.val_results, class_names=class_names
            )

        pruned_soup_params = {}
        candidate_model_paths = [candidate.model_path for candidate in current_candidates]
        for i, model_path in enumerate(candidate_model_paths):
            logger.info(f'[{i + 1}/{num_models}] Adding model {model_path} to the starting soup')
            model_state_dict = torch.load(model_path, map_location=device)['model_state_dict']
            if i == 0:
                pruned_soup_params = {
                    k: v.clone() * (1.0 / num_models) for k, v in model_state_dict.items()
                }
            else:
                pruned_soup_params = {
                    k: v.clone() * (1.0 / num_models) + pruned_soup_params[k]
                    for k, v in model_state_dict.items()
                }

        # Compute the baseline performance using validation set
        pruned_soup_ingredients = [candidate.model_path for candidate in current_candidates]
        model.load_state_dict(pruned_soup_params)
        best_val_result_so_far = eval_model(
            model=model,
            eval_data_loader=val_data_loader,
            device=device,
            num_classes=num_classes,
        )[comp_metric]

        for iteration in range(args.pruned_soup_num_iters):
            if len(pruned_soup_ingredients) < 2:
                logger.info('Not enough ingredients to prune')
                break

            any_improvement = False

            # Try removing each candidate from worst to best
            candidates_to_try = list(reversed(current_candidates))  # Start from worst performers

            for candidate_to_remove in candidates_to_try:
                # Skip if this candidate is not in the current soup
                if candidate_to_remove.model_path not in pruned_soup_ingredients:
                    continue

                logger.info(
                    f'[Iter {iteration + 1}/{args.pruned_soup_num_iters}] Trying removing {candidate_to_remove.model_path}'
                )

                # get the potential new soup by removing this worst model
                worst_ingredient_params = torch.load(
                    candidate_to_remove.model_path,
                    map_location=device,
                )['model_state_dict']
                num_ingredients = len(pruned_soup_ingredients)
                assert num_ingredients >= 2

                potential_new_pruned_soup_params = remove_ingredient_from_soup(
                    soup=pruned_soup_params,
                    num_ingredients=num_ingredients,
                    ingredient=worst_ingredient_params,
                )

                # test the new-branch model
                model.load_state_dict(potential_new_pruned_soup_params)
                cur_val_result = eval_model(
                    model=model,
                    eval_data_loader=val_data_loader,
                    device=device,
                    num_classes=num_classes,
                )[comp_metric]
                if comp_metric == 'loss':
                    cur_val_result = -cur_val_result

                # if `comp_metric` improves, remove the model from the soup
                logger.info(
                    f'[Iter {iteration + 1}/{args.pruned_soup_num_iters}] Potential new pruned soup val {comp_metric} {abs(cur_val_result):0.6f}, '
                    f'best so far {abs(best_val_result_so_far):0.6f}.'
                )
                if cur_val_result + EPS > best_val_result_so_far:
                    pruned_soup_ingredients.remove(candidate_to_remove.model_path)
                    current_candidates.remove(candidate_to_remove)
                    best_val_result_so_far = cur_val_result
                    pruned_soup_params = potential_new_pruned_soup_params
                    logger.info(
                        f'[Iter {iteration + 1}/{args.pruned_soup_num_iters}] Removed model {candidate_to_remove.model_path} from pruned soup'
                    )
                    any_improvement = True
                    break  # Only remove one model per iteration

            if not any_improvement:
                logger.info(f'No improvement found in iteration {iteration + 1}')
                break

        result_data['ingredients'] = pruned_soup_ingredients
        result_data['num_ingredients'] = len(pruned_soup_ingredients)
        result_data[f'best_val_{comp_metric}'] = abs(best_val_result_so_far)

        # test the final pruned soup
        print('** Pruned soup ingredients: **')
        for ingredient in pruned_soup_ingredients:
            print(f'  - {ingredient}')

        model.load_state_dict(pruned_soup_params)
        test_results = eval_model(
            model=model,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        print('** Pruned soup test results: **')
        print_eval_results(eval_results=test_results, prefix='test')

        # save the results
        result_data['pruned_soup'] = convert_eval_results_to_dict(
            eval_results=test_results, class_names=class_names
        )
        with open(pruned_soup_result_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        logger.info(f'Pruned soup results saved to {pruned_soup_result_file}')

        # save the soup model
        pruned_soup_model_path = os.path.join(args.output_dir, 'pruned_soup.pth')
        torch.save(
            {
                'model_state_dict': pruned_soup_params,
                'test_results': test_results,
            },
            pruned_soup_model_path,
        )


def add_ingredient_to_soup(
    soup: dict[str, Any], num_ingredients: int, ingredient: dict[str, Any]
) -> dict[str, Any]:
    if num_ingredients <= 0:
        return ingredient
    new_soup = {
        k: (soup[k].clone() * num_ingredients + ingredient[k].clone()) / (num_ingredients + 1)
        for k in ingredient
    }
    return new_soup


def remove_ingredient_from_soup(
    soup: dict[str, Any], num_ingredients: int, ingredient: dict[str, Any]
) -> dict[str, Any]:
    if num_ingredients <= 1:
        raise ValueError('Cannot remove ingredient from soup with less than 2 ingredients.')
    new_soup = {
        k: (soup[k].clone() * num_ingredients - ingredient[k].clone()) / (num_ingredients - 1)
        for k in ingredient
    }
    return new_soup


def main():
    parser = argparse.ArgumentParser(
        description='Run test with model soups',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_test_with_model_soups_opts(parser)
    args = parser.parse_args()

    test_with_model_soups(args)


if __name__ == '__main__':
    main()
