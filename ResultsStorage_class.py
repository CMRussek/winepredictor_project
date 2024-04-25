import json
import sqlite3


class ResultsStorage:
    def __init__(self, db_name='wine_classification_results.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def create_table(self):
        schema = '''CREATE TABLE IF NOT EXISTS results
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     model TEXT,
                     type TEXT,
                     result TEXT)'''
        self.cursor.execute(schema)
        self.conn.commit()

    def insert_result(self, model, result_type, result):
        if isinstance(result, dict):
            result = json.dumps(result)
        self.cursor.execute('''INSERT INTO results (model, type, result)
                               VALUES (?, ?, ?)''', (model, result_type, result))
        self.conn.commit()

    def close_connection(self):
        self.conn.close()
