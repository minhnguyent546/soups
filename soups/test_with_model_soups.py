"""Run test with model soups"""

import argparse
import os
import json
from dataclasses import dataclass

import torch
import torchvision
from torch.utils.data import DataLoader

import soups.utils as utils
from soups.opts import add_test_with_model_soups_opts
from soups.utils.logger import logger, init_logger
from soups.utils.training import (
    eval_model,
    EvalResults,
    make_model,
    print_eval_results,
)


@dataclass
class Candidate:
    model_path: str
    eval_results: EvalResults

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

    # find all model checkpoint files
    model_paths: list[str] = []
    for model_path in args.checkpoint_path:
        if os.path.isfile(model_path):
            model_paths.append(model_path)
        elif os.path.isdir(model_path):
            model_paths.extend(
                os.path.join(model_path, f)
                for f in os.listdir(model_path) if f.endswith('.pth')
            )
    model_paths = list(set(model_paths))  # remove duplicates
    if not model_paths:
        logger.error('No model checkpoints found.')
        exit(1)
    num_models = len(model_paths)
    logger.info(f'Found total {len(model_paths)} model checkpoints')

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
    classes = test_dataset.classes
    num_classes = len(classes)

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
    for model_path in model_paths:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f'Evaluating model: {model_path}')
        val_results = eval_model(
            model=model,
            eval_data_loader=val_data_loader,
            device=device,
            num_classes=num_classes,
        )
        candidates.append(Candidate(model_path, val_results))
        print_eval_results(val_results, prefix='val')

    # uniform soup (i.e. mixing all models together)
    if args.uniform_soup:
        logger.info('** Start cooking uniform soup **')
        uniform_soup_result_file = os.path.join(
            args.output_dir,
            'uniform_soup_results.json',
        )

        result_data = {}
        result_data['models'] = {}
        for candidate in candidates:
            result_data['models'][candidate.model_path] = candidate.eval_results

        uniform_soup_params = {}
        for i, model_path in enumerate(model_paths):
            logger.info(f'[{i + 1}/{num_models}] Adding model {model_path} to uniform soup')
            model_state_dict = torch.load(model_path, map_location=device)['model_state_dict']
            if i == 0:
                uniform_soup_params = {
                    k: v.clone() * (1.0 / num_models)
                    for k, v in model_state_dict.items()
                }
            else:
                uniform_soup_params = {
                    k: v.clone() * (1.0 / num_models) + uniform_soup_params[k]
                    for k, v in model_state_dict.items()
                }

        # test the uniform soup
        model = make_model(
            model_name=args.model,
            num_classes=num_classes,
        ).to(device)
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
        result_data['uniform_soup'] = test_results
        with open(uniform_soup_result_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        logger.info(f'Uniform soup results saved to {uniform_soup_result_file}')

        # save the soup model
        uniform_soup_model_path = os.path.join(args.output_dir, 'uniform_soup.pth')
        torch.save({
            'model_state_dict': uniform_soup_params,
            'test_results': test_results,
        }, uniform_soup_model_path)

    if args.greedy_soup:
        logger.info('** Start cooking greedy soup **')
        greedy_soup_result_file = os.path.join(
            args.output_dir,
            'greedy_soup_results.json',
        )
        # sort models by decreasing test accuracy
        candidates = sorted(candidates, key=lambda item: item.eval_results['accuracy'], reverse=True)

        result_data = {}
        result_data['models'] = {}
        for candidate in candidates:
            result_data['models'][candidate.model_path] = candidate.eval_results

        # start the soup by using the first ingredient.
        greedy_soup_ingredients = [candidates[0].model_path]
        greedy_soup_params = torch.load(
            candidates[0].model_path,
            map_location=device,
        )['model_state_dict']
        best_val_acc_so_far = candidates[0].eval_results['accuracy']

        for i in range(1, num_models):
            logger.info(f'Trying model {candidates[i].model_path}')

            # get the potential new soup by adding the current model
            new_ingredient_params = torch.load(
                candidates[i].model_path,
                map_location=device,
            )['model_state_dict']
            num_ingredients = len(greedy_soup_ingredients)
            potential_greedy_soup_params = {
                k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) +
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }

            # test the new-branch model
            model = make_model(model_name=args.model, num_classes=num_classes).to(device)
            model.load_state_dict(potential_greedy_soup_params)
            cur_val_accuracy = eval_model(
                model=model,
                eval_data_loader=val_data_loader,
                device=device,
                num_classes=num_classes,
            )['accuracy']

            # if accuracy improves, add the model to the soup
            logger.info(
                f'Potential greedy soup val acc {cur_val_accuracy:0.6f}, '
                f'best so far {best_val_acc_so_far:0.6f}.'
            )
            if cur_val_accuracy > best_val_acc_so_far:
                greedy_soup_ingredients.append(candidates[i].model_path)
                best_val_acc_so_far = cur_val_accuracy
                greedy_soup_params = potential_greedy_soup_params
                logger.info(f'Added model {candidates[i].model_path} to greedy soup')

        result_data['ingredients'] = greedy_soup_ingredients
        result_data['best_val_accuracy'] = best_val_acc_so_far

        # test the final greedy soup
        print('** Greedy soup ingredients: **')
        for ingredient in greedy_soup_ingredients:
            print(f'  - {ingredient}')

        model = make_model(
            model_name=args.model,
            num_classes=num_classes,
        ).to(device)
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
        result_data['greedy_soup'] = test_results
        with open(greedy_soup_result_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        logger.info(f'Uniform soup results saved to {greedy_soup_result_file}')

        # save the soup model
        greedy_soup_model_path = os.path.join(args.output_dir, 'greedy_soup.pth')
        torch.save({
            'model_state_dict': greedy_soup_params,
            'test_results': test_results,
        }, greedy_soup_model_path)

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
