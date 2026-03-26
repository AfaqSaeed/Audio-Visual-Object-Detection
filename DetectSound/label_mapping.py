from __future__ import annotations

import csv
import re
from pathlib import Path


def normalize_label(label: str) -> str:
    label = label.strip().casefold()
    label = label.replace("&", " and ")
    label = re.sub(r"[^\w\s/,-]", " ", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip()


def load_label_to_id(metadata_csv: Path) -> dict[str, int]:
    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["display_name"]: int(row["index"]) for row in reader}


EXACT_AUDIOSET_TO_EPIC: dict[str, str] = {
    "speech": "human",
    "male speech, man speaking": "human",
    "female speech, woman speaking": "human",
    "child speech, kid speaking": "human",
    "conversation": "human",
    "narration, monologue": "human",
    "babbling": "human",
    "whispering": "human",
    "yell": "human",
    "shout": "human",
    "bellow": "human",
    "whoop": "human",
    "scream": "human",
    "sigh": "human",
    "groan": "human",
    "grunt": "human",
    "singing": "human",
    "breathing": "human",
    "snoring": "human",
    "gasp": "human",
    "cough": "human",
    "throat clearing": "human",
    "sneeze": "human",
    "sniff": "human",
    "chewing, mastication": "drink / eat",
    "bite": "drink / eat",
    "gulp": "drink / eat",
    "burping, eructation": "drink / eat",
    "fart": "human",
    "walk, footsteps": "footstep",
    "run": "footstep",
    "shuffle": "footstep",
    "zipper (clothing)": "zip",
    "coin (dropping)": "metal-only collision",
    "chink, clink": "metal-only collision",
    "water": "water",
    "water tap, faucet": "water",
    "sink (filling or washing)": "water",
    "drip": "water",
    "trickle, dribble": "water",
    "pour": "pour",
    "spray": "spray",
    "sizzle": "sizzling / boiling",
    "boiling": "sizzling / boiling",
    "stir": "stir / mix / whisk",
    "scrape": "scrub / scrape / scour / wipe",
    "rub": "scrub / scrape / scour / wipe",
    "filing (rasp)": "scrub / scrape / scour / wipe",
    "clicking": "click",
    "typewriter": "click",
    "computer keyboard": "click",
    "typewriter keyboard": "click",
    "beep, bleep": "beep",
    "inside, small room": "background",
    "silence": "background",
    "pink noise": "background",
    "white noise": "background",
    "vacuum cleaner": "hoover / fan",
    "fan": "hoover / fan",
    "mechanical fan": "hoover / fan",
    "blender": "kettle / mixer / appliance",
    "food processor": "kettle / mixer / appliance",
    "microwave oven": "kettle / mixer / appliance",
    "kettle whistle": "kettle / mixer / appliance",
    "scissors": "cut / chop",
    "chopping (food)": "cut / chop",
    "sliding door": "open / close",
    "door": "open / close",
    "drawer open or close": "open / close",
    "cupboard open or close": "open / close",
    "paper": "rustle",
    "crumpling, crinkling": "rustle",
    "crinkling": "rustle",
}


KEYWORD_AUDIOSET_TO_EPIC: list[tuple[tuple[str, ...], str]] = [
    (("speech", "talk", "conversation", "whisper", "scream", "laugh", "cough", "sneeze"), "human"),
    (("footstep", "footsteps", "walk", "run", "shuffle"), "footstep"),
    (("water tap", "faucet", "sink", "drip", "splash", "water"), "water"),
    (("pour", "liquid"), "pour"),
    (("spray", "aerosol"), "spray"),
    (("zip", "zipper"), "zip"),
    (("beep", "bleep"), "beep"),
    (("click", "keyboard", "mouse"), "click"),
    (("sizzle", "boil"), "sizzling / boiling"),
    (("stir", "whisk", "mix"), "stir / mix / whisk"),
    (("scrape", "scour", "wipe", "rub", "file"), "scrub / scrape / scour / wipe"),
    (("vacuum", "fan"), "hoover / fan"),
    (("blender", "mixer", "kettle", "microwave", "appliance"), "kettle / mixer / appliance"),
    (("cut", "chop", "slice", "scissors"), "cut / chop"),
    (("open", "close", "door", "drawer", "cupboard"), "open / close"),
    (("paper", "rustle", "crumple", "crinkle"), "rustle"),
    (("chew", "bite", "gulp", "eat", "drink"), "drink / eat"),
    (("knead", "dough"), "kneading"),
    (("slide", "drag"), "slide object"),
    (("clink", "chink", "cutlery", "silverware", "coin"), "metal-only collision"),
    (("glass", "bottle"), "glass-only collision"),
    (("ceramic", "plate", "dish"), "ceramic-only collision"),
    (("wood", "wooden"), "wood-only collision"),
]


def map_audioset_label_to_epic_label(audioset_label: str) -> str | None:
    normalized = normalize_label(audioset_label)

    exact = EXACT_AUDIOSET_TO_EPIC.get(normalized)
    if exact is not None:
        return exact

    for keywords, epic_label in KEYWORD_AUDIOSET_TO_EPIC:
        if any(keyword in normalized for keyword in keywords):
            return epic_label

    return None


def map_audioset_label_to_epic(
    audioset_label: str,
    epic_label_to_id: dict[str, int],
) -> tuple[str, int] | None:
    epic_label = map_audioset_label_to_epic_label(audioset_label)
    if epic_label is None:
        return None

    epic_id = epic_label_to_id.get(epic_label)
    if epic_id is None:
        return None

    return epic_label, epic_id
