"""Tests for XML-based ReAct parser."""

import pytest
from miiflow_llm.core.react.parsing import XMLReActParser


class TestXMLParser:
    """Test XML parser with various input formats."""

    def test_parse_tool_call_clean(self):
        """Test parsing valid tool call XML."""
        parser = XMLReActParser()
        response = """
        <thinking>I need to search for weather information in Paris</thinking>
        <tool_call name="weather">{"city": "Paris", "country": "France"}</tool_call>
        """

        result = parser.parse_complete(response)

        assert result["thought"] == "I need to search for weather information in Paris"
        assert result["action_type"] == "tool_call"
        assert result["action"] == "weather"
        assert result["action_input"] == {"city": "Paris", "country": "France"}

    def test_parse_final_answer_clean(self):
        """Test parsing final answer XML."""
        parser = XMLReActParser()
        response = """
        <thinking>I have enough information to provide a complete answer</thinking>
        <answer>The weather in Paris is currently 22°C and partly cloudy.</answer>
        """

        result = parser.parse_complete(response)

        assert result["thought"] == "I have enough information to provide a complete answer"
        assert result["action_type"] == "final_answer"
        assert result["answer"] == "The weather in Paris is currently 22°C and partly cloudy."

    def test_parse_tool_call_with_markdown(self):
        """Test parsing XML with markdown wrapper."""
        parser = XMLReActParser()
        response = """
        <thinking>I need to calculate distance</thinking>
        <tool_call name="distance_calculator">{"from": "NYC", "to": "LA"}</tool_call>
        """

        result = parser.parse_complete(response)

        assert result["action_type"] == "tool_call"
        assert result["action"] == "distance_calculator"

    def test_parse_unclosed_thinking_tag(self):
        """Test parsing with unclosed thinking tag."""
        parser = XMLReActParser()
        response = """
        <thinking>I need to search for information</thinking>
        <tool_call name="search">{"query": "test"}</tool_call>
        """

        result = parser.parse_complete(response)

        assert result["action_type"] == "tool_call"

    def test_parse_malformed_json_input(self):
        """Test handling of malformed JSON in input tag."""
        parser = XMLReActParser()
        response = """
        <thinking>Need to call tool</thinking>
        <tool_call name="search">just a string, not JSON</tool_call>
        """

        result = parser.parse_complete(response)

        assert result["action_type"] == "tool_call"
        assert result["action"] == "search"
        assert result["action_input"] == {}

    def test_parse_missing_thinking_tag(self):
        """Test error when thinking tag is missing."""
        parser = XMLReActParser()
        response = """
        <tool_call name="search">{"query": "test"}</tool_call>
        """

        with pytest.raises(ValueError, match="No valid ReAct XML tags found"):
            parser.parse_complete(response)

    def test_parse_missing_tool_name(self):
        """Test valid parsing even without tool name attribute."""
        parser = XMLReActParser()
        response = """
        <thinking>Need to search</thinking>
        <answer>I have completed the task.</answer>
        """

        result = parser.parse_complete(response)
        assert result["action_type"] == "final_answer"

    def test_parse_empty_answer(self):
        """Test parsing with empty answer tag."""
        parser = XMLReActParser()
        response = """
        <thinking>Done thinking</thinking>
        <answer></answer>
        """

        result = parser.parse_complete(response)
        assert result["action_type"] == "final_answer"
        assert result["answer"] == ""

    def test_parse_mixed_content_extraction(self):
        """Test extraction from mixed natural language and XML."""
        parser = XMLReActParser()
        response = """
        Let me help you with that.

        <thinking>I need to search for this information</thinking>
        <tool_call name="search">{"query": "test"}</tool_call>

        I'll execute that now.
        """

        result = parser.parse_complete(response)

        assert result["action_type"] == "tool_call"
        assert result["action"] == "search"

    def test_parse_case_insensitive(self):
        """Test that parser is case-insensitive for tags."""
        parser = XMLReActParser()
        response = """
        <THINKING>Need to search</THINKING>
        <TOOL_CALL name="search">{"query": "test"}</TOOL_CALL>
        """

        result = parser.parse_complete(response)

        assert result["action_type"] == "tool_call"
        assert result["action"] == "search"

    def test_parse_multiline_answer(self):
        """Test parsing answer with multiple lines."""
        parser = XMLReActParser()
        response = """
        <thinking>I have the information needed</thinking>
        <answer>
        Here is a comprehensive answer:
        • Point 1: First detail
        • Point 2: Second detail
        • Point 3: Third detail

        This covers everything.
        </answer>
        """

        result = parser.parse_complete(response)

        assert result["action_type"] == "final_answer"
        assert "Point 1" in result["answer"]
        assert "Point 2" in result["answer"]
        assert "Point 3" in result["answer"]

    def test_parse_complex_nested_json(self):
        """Test parsing tool input with nested JSON."""
        parser = XMLReActParser()
        response = """
        <thinking>Need to execute complex query</thinking>
        <tool_call name="database_query">{
                "table": "users",
                "filters": {
                    "age": {"gt": 18, "lt": 65},
                    "active": true
                },
                "sort": ["name", "asc"]
            }</tool_call>
        """

        result = parser.parse_complete(response)

        assert result["action_type"] == "tool_call"
        assert result["action"] == "database_query"
        assert result["action_input"]["table"] == "users"
        assert result["action_input"]["filters"]["age"]["gt"] == 18
        assert result["action_input"]["filters"]["active"] is True


class TestXMLParserEdgeCases:
    """Test edge cases and error handling."""

    def test_completely_invalid_xml(self):
        """Test that completely invalid input raises error."""
        parser = XMLReActParser()
        response = "This is just plain text with no XML tags at all"

        with pytest.raises(ValueError):
            parser.parse_complete(response)

    def test_partial_xml_fragments(self):
        """Test handling of incomplete XML fragments."""
        parser = XMLReActParser()
        response = "<thinking>I need to"

        with pytest.raises(ValueError):
            parser.parse_complete(response)

    def test_get_parser_name(self):
        """Test parser class name."""
        parser = XMLReActParser()
        assert parser.__class__.__name__ == "XMLReActParser"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
