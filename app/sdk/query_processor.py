class QueryProcessor:
    def __init__(self, queries):
        self.queries = queries

    def get_query(self, qid):
        for query in self.queries:
            if query["id"].upper() == qid.upper():
                return query["text"]
        return None
