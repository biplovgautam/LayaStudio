"""The registry card keeps the Hub card's measurements and points its examples at the registry."""

from layastudio.publish_systemone import QUESTIONS_MARK, registry_card

HUB_CARD = """---
license: apache-2.0
---

# Snake

## Measured on the held-out test split

| Accuracy | 98.8% |

## Use it

```bash
pip install laya-mlx
```

```python
agent = laya.load("me/laya-snake-mlx")
```

Ask it **these** questions: the wording matters.

## Provenance
"""


def test_use_it_section_is_replaced_and_the_rest_kept():
    card = registry_card(HUB_CARD, "me/laya-snake")
    assert "systemone pull me/laya-snake" in card
    assert 'snapshot_download("me/laya-snake")' in card
    assert 'laya.load("me/laya-snake-mlx")' not in card
    assert "| Accuracy | 98.8% |" in card
    assert QUESTIONS_MARK in card
    assert "## Provenance" in card


def test_a_card_without_the_markers_is_left_alone():
    assert registry_card("# Just a title\n", "me/x") == "# Just a title\n"
