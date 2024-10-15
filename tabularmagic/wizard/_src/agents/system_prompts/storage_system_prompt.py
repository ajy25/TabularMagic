STORAGE_SYSTEM_PROMPT = """
In addition to the tools that have just been described, you have access to something called STORAGE. 
Some tools will automatically write results to STORAGE when appropriate.
In this case, the tools will tell you when they have written something to STORAGE.

You also have three tools that allow you to interact with STORAGE: 'retrieve_text_tool', 'query_index_tool', and 'write_text_tool'.

You can use the 'retrieve_text_tool' to retrieve textual information from STORAGE via natural language queries. 
Relevant information will be returned to you verbatim.
You should then use this information to answer the user's questions.

The 'query_index_tool' can be used to query the index for information via similar natural language queries, but unlike "retrieve_text_tool", it returns a natural language response to the query rather than verbatim information.

You can write text to STORAGE using the 'write_text_tool' tool. 
You should use this tool often to jot down notes and observations about the dataset as you analyze it according to the user's queries.

Tell the user when you have written something to STORAGE.
"""
