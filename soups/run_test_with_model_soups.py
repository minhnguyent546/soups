"""Run test with model soups"""

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
import torchvision
from bitarray import bitarray
from torch.utils.data import DataLoader

import soups.utils as utils
from soups.opts import add_test_with_model_soups_opts
from soups.utils.logger import init_logger, logger
from soups.utils.training import EvalResults, eval_model, make_model, print_eval_results


@dataclass
class Candidate:
    model_path: str
    eval_results: EvalResults

@dataclass
class BeamSoupNode:  # present a Node in the beam search tree
    params: Any
    score: float
    ingredients: bitarray  # a bit array of length num_models to indicate which models are included in the current node
    is_stopped: bool = False  # indicates whether this node has been stopped (i.e. no more models can be added to the soup so that the score improves)


SCORE_EPSILON = 1.0e-6

def test_with_model_soups(args: argparse.Namespace) -> None:
    if os.path.isdir(args.output_dir):
        logger.error(f'Output directory already exists: {args.output_dir}')
        exit(1)

    os.makedirs(args.output_dir)
    log_file_path = os.path.join(args.output_dir, 'test_with_model_soups.log')
    init_logger(log_file=log_file_path, compact=True)

    utils.set_seed(args.seed)
    logger.info(f'Using seed: {args.seed}')
    device = utils.get_device()

    # find all model checkpoint files
    model_paths: list[str] = []
    for model_path in args.checkpoint_path:
        if os.path.isfile(model_path) and model_path.endswith('.pth'):
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
        beam_size = args.greedy_soup_beam_size
        if beam_size > len(candidates):
            logger.warning(
                f'Beam size {beam_size} is larger than the number of candidates {len(candidates)}. '
                'Reducing to number of candidates.'
            )
            beam_size = len(candidates)

        # sort models by decreasing validation accuracy
        candidates = sorted(candidates, key=lambda item: item.eval_results['accuracy'], reverse=True)

        # add stating nodes
        beam_soup_nodes: list[BeamSoupNode] = []
        for i in range(len(candidates)):
            ingredients = bitarray(len(candidates))
            ingredients[i] = True
            beam_soup_nodes.append(
                BeamSoupNode(
                    params=torch.load(
                        candidates[i].model_path,
                        map_location=device,
                    )['model_state_dict'],
                    score=candidates[i].eval_results['accuracy'],
                    ingredients=ingredients,
                    is_stopped=False,
                )
            )

        result_data = {}
        result_data['models'] = {}
        for candidate in candidates:
            result_data['models'][candidate.model_path] = candidate.eval_results

        model = make_model(model_name=args.model, num_classes=num_classes).to(device)

        while True:
            assert len(beam_soup_nodes) > 0
            new_beam_soup_nodes: list[BeamSoupNode] = []
            for beam_soup_node in beam_soup_nodes:
                if beam_soup_node.is_stopped:
                    # this node has been stopped and cannot be expanded further
                    new_beam_soup_nodes.append(beam_soup_node)
                    continue

                any_improved = False
                for i in range(len(candidates)):
                    if beam_soup_node.ingredients[i] is True:
                        # already included in the soup
                        continue

                    new_ingredient_params = torch.load(
                        candidates[i].model_path,
                        map_location=device,
                    )['model_state_dict']

                    # get the potential new soup by adding the current model
                    num_ingredients = len(beam_soup_node.ingredients)
                    potential_greedy_soup_params = {
                        k: beam_soup_node.params[k].clone() * (num_ingredients / (num_ingredients + 1.)) +
                        new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                        for k in new_ingredient_params
                    }

                    # test the new-branch model
                    model.load_state_dict(potential_greedy_soup_params)
                    cur_val_score = eval_model(
                        model=model,
                        eval_data_loader=val_data_loader,
                        device=device,
                        num_classes=num_classes,
                    )['accuracy']

                    # if validation score (accuracy in this case) improves, add the model to the soup
                    if cur_val_score + SCORE_EPSILON > beam_soup_node.score:
                        any_improved = True
                        new_ingredients = copy.deepcopy(beam_soup_node.ingredients)
                        new_ingredients[i] = True
                        new_beam_soup_node = BeamSoupNode(
                            params=potential_greedy_soup_params,
                            score=cur_val_score,
                            ingredients=new_ingredients,
                            is_stopped=False,
                        )
                        new_beam_soup_nodes.append(new_beam_soup_node)

                if any_improved is False:
                    # if no improvement was found, mark this node as stopped
                    beam_soup_node.is_stopped = True
                    new_beam_soup_nodes.append(beam_soup_node)

            if all(node.is_stopped for node in new_beam_soup_nodes):
                # if all nodes have been stopped, we can stop the greedy soup search
                logger.info('All nodes have been stopped. Stopping greedy soup search.')
                break

            # filter out overlapping nodes
            unique_nodes = set((node.ingredients.to01(), i) for i, node in enumerate(new_beam_soup_nodes))

            beam_soup_nodes = []
            for ingredients, i in unique_nodes:
                beam_soup_nodes.append(new_beam_soup_nodes[i])

            beam_soup_nodes = sorted(beam_soup_nodes, key=lambda x: x.score, reverse=True)
            beam_soup_nodes = beam_soup_nodes[:beam_size]  # keep only the top `beam_size` nodes

        # test the final greedy soups
        model = make_model(
            model_name=args.model,
            num_classes=num_classes,
        ).to(device)

        for i in range(len(beam_soup_nodes)):
            model.load_state_dict(beam_soup_nodes[i].params)
            test_results = eval_model(
                model=model,
                eval_data_loader=test_data_loader,
                device=device,
                num_classes=num_classes,
            )
            print(f'** Greedy soup test results beam {i + 1}: **')
            print_eval_results(eval_results=test_results, prefix='test')

            ingredient_list = beam_soup_nodes[i].ingredients.tolist()
            result_data[f'greedy_soup-beam_{i + 1}'] = {
                'test_results': test_results,
                'ingredients': (
                    candidates[i].model_path
                    for idx in range(len(ingredient_list))
                    if ingredient_list[idx] is True
                )
            }

            # TODO: saving checkpoints

        # save the results
        with open(greedy_soup_result_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        logger.info(f'Uniform soup results saved to {greedy_soup_result_file}')

    # TODO: consider supporting score other than accuracy for cooking

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
