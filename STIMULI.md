# Stimuli: sources and licences

The 54 stimuli are three image sets of 18 images each (`resources/{action,greebles,taxonomic}_info.csv`).
They are in `experiment/images`; `experiment_mid_lab/images` is a copy for the second lab setup.
The image files keep the licences of their sources listed below; they are not covered by the
licences of this repository.

## THINGS images (29 images, not included)

Twenty-nine natural images were taken from the THINGS database:

> Hebart, M. N., Dickter, A. H., Kidder, A., Kwok, W. Y., Corriveau, A., Van Wicklin, C., &
> Baker, C. I. (2019). THINGS: A database of 1,854 object concepts and more than 26,000
> naturalistic object images. *PLOS ONE*, 14(10), e0223792.
> <https://doi.org/10.1371/journal.pone.0223792>

THINGS provides its images for non-commercial research only, and its licence does not allow
redistributing them. They are therefore not part of this repository.
`resources/things_images.csv` lists them with their path in the THINGS archive and the
SHA-256 checksum of the file used in the experiment; `scripts/restore_things_images.py`
copies them from your own download of `images_THINGS.zip` (<https://osf.io/jum2f/>) into the
experiment folders (see the README).

Taxonomic set: `bee_01b`, `bee_10s`, `branch_09s`, `ladybug_02s`, `ladybug_07s`, `leaf_02s`,
`leaf_06s`, `leaf_07s`, `tree_07s`, `trunk_06s`, `wasp_03s`, `wasp_06s`.

Action set: `coin_12s`, `coin_14s`, `key_03s`, `key_07s`, `key_08s`, `key_14s`,
`letter_opener_01b`, `letter_opener_06s`, `letter_opener_10s`, `match_01b`, `match_04s`,
`matchbox_01b`, `screwdriver_01s`, `screwdriver_03s`, `screwdriver_04s`, `screwdriver_12s`,
`screwdriver_13s`.

## Images from the internet (6 images, taxonomic set)

| File | Title and author | Licence |
|---|---|---|
| `katydid_07.jpg` | "Systella dusmeti? - Khao Sok National Park" by Rushen! | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) |
| `katydid_10.jpg` | "False-leaf katydid Aegimia sp (possibly A. elongata)" by Dis da fi we | [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/) |
| `phyllium_05.jpg` | "Baby leaf insect" by pixelthing | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) |
| `phyllium_14.jpg` | "Das Große Wandelnde Blatt" by vobebis | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `stick_insect_01.jpg` | "Moss mimic stick insect, Trychopeplus thaumasius?" by In Memoriam: Ecuador Megadiverso | [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/) |
| `stick_insect_09.jpg` | "Moss-camouflaged stick insect" by Pasha Kirillov | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) |

The files in this repository are the versions shown in the experiment and may be cropped or
resized versions of the originals; they are shared under the licence of their original.

## Own photograph (1 image, action set)

`coin_additional.jpg` was taken by the authors and is licensed under CC BY 4.0, as the data
of this repository.

## Greebles (18 images used, 36 files)

Greebles (Gauthier & Tarr, 1997), families 1, 3 and 5, from the stimulus collection of the
Tarr lab (<https://sites.google.com/andrew.cmu.edu/tarrlab/stimuli>).

> Stimulus images courtesy of Michael J. Tarr, Carnegie Mellon University, http://www.tarrlab.org/

They are used under the Tarr lab's terms: non-commercial use only, under the
[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0/)
licence. The folder holds two views (`_v1`, `_v2`) of 18 individuals; the 18 images used in
the experiment are listed in `resources/greebles_info.csv`.
