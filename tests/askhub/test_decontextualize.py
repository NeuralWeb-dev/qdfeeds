"""A follow-up is searched as the model's standalone rewrite whenever the rewrite differs."""

import pytest

from askhub.models import SearchRequest
from askhub.pre_retrieval import decontextualize

FIRST = ("Something vegetarian and Indian for a cold winter night, highly rated, that doesn't "
         "need a long list of whole spices I'd have to go out and buy.")


class Replies:
    def __init__(self, output):
        self.output = output

    async def structured(self, instruction, payload, **kwargs):
        return self.output


def follow_up(query="I am vegetarian"):
    return SearchRequest(query=query, previous_queries=(FIRST,))


@pytest.mark.parametrize("changed", [True, False, None])
async def test_a_rewrite_is_used_whatever_the_model_says_about_it(changed):
    rewrite = "A highly rated vegetarian Indian dish for winter that needs few whole spices."
    output = {"standalone_query": rewrite}
    if changed is not None:
        output["changed"] = changed
    assert await decontextualize(follow_up(), Replies(output), 5) == rewrite


async def test_a_message_returned_as_it_was_is_searched_as_it_was():
    output = {"standalone_query": " pancakes ", "changed": True}
    assert await decontextualize(follow_up("pancakes"), Replies(output), 5) is None


async def test_a_provider_failure_searches_the_original_query():
    class Broken:
        async def structured(self, *args, **kwargs):
            raise RuntimeError("down")
    assert await decontextualize(follow_up(), Broken(), 5) is None
