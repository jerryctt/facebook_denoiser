# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
import sys

import torch
import torchaudio

from .audio import Audioset, find_audio_files
from . import distrib, pretrained
from .demucs import DemucsStreamer
from torch.utils.mobile_optimizer import optimize_for_mobile
from .utils import LogProgress

logger = logging.getLogger(__name__)


def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    pretrained.add_model_flags(parser)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--streaming', action="store_true", default=True,
                        help="true streaming evaluation for Demucs")


parser = argparse.ArgumentParser(
        'denoiser.enhance',
        description="Speech enhancement using Demucs - Generate enhanced files")
add_flags(parser)
parser.add_argument("--out_dir", type=str, default="./out",
                    help="directory putting enhanced wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")

group = parser.add_mutually_exclusive_group()
group.add_argument("--noisy_dir", type=str, default="./in",
                   help="directory including noisy wav files")
group.add_argument("--noisy_json", type=str, default=None,
                   help="json file including noisy wav files")


def get_estimate(model, noisy, args):
    torch.set_num_threads(1)
    if args.streaming:
        streamer = DemucsStreamer(model, dry=args.dry)
        with torch.no_grad():
            estimate = torch.cat([
                streamer.feed(noisy[0]),
                streamer.flush()], dim=1)[None]
    else:
        with torch.no_grad():
            estimate = model(noisy)
            estimate = (1 - args.dry) * estimate + args.dry * noisy
    return estimate


def save_wavs(estimates, noisy_sigs, filenames, out_dir, sr=16_000):
    # Write result
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        write(noisy, filename + "_noisy.wav", sr=sr)
        write(estimate, filename + "_enhanced.wav", sr=sr)


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def get_dataset(args, sample_rate, channels):
    if hasattr(args, 'dset'):
        paths = args.dset
    else:
        paths = args
    if paths.noisy_json:
        with open(paths.noisy_json) as f:
            files = json.load(f)
    elif paths.noisy_dir:
        files = find_audio_files(paths.noisy_dir)
    else:
        logger.warning(
            "Small sample set was not provided by either noisy_dir or noisy_json. "
            "Skipping enhancement.")
        return None
    return Audioset(files, with_path=True,
                    sample_rate=sample_rate, channels=channels, convert=True)


def _estimate_and_save(model, noisy_signals, filenames, out_dir, args):
    estimate = get_estimate(model, noisy_signals, args)
    save_wavs(estimate, noisy_signals, filenames, out_dir, sr=model.sample_rate)


def enhance(args, model=None, local_out_dir=None):
    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()
    example = torch.rand(1, 597)
    streamer = DemucsStreamer(model, dry=1)
    traced_script_module = torch.jit.script( streamer.feed ) #, example )

#    traced_script_module = torch.jit.trace(model, example)
    optimized_model = optimize_for_mobile(traced_script_module, backend='metal')
    #m = torch.jit.script(model, example)
    print(torch.jit.export_opnames(optimized_model))
    optimized_model._save_for_lite_interpreter('sss.pt')



if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    enhance(args, local_out_dir=args.out_dir)
