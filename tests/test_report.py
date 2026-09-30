from string import Template

from fraudops.report import TEMPLATE, values


def test_every_number_of_the_page_comes_from_the_published_results():
    page = Template(TEMPLATE.read_text()).substitute(values())
    assert "$" not in page
    assert "k€" in page
