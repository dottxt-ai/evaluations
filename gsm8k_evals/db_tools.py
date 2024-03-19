import sqlite3
from datetime import datetime

def create_evaluation_table(db):
    with sqlite3.connect(db) as con:
        cur = con.cursor()
        result = cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluations(model, dataset, sub_set, 
        start_time, end_time, cot, n_shot, sampler, n_samples, prompt_name, struct_name);
        """).fetchall()
        con.commit()
    return result

def create_result_table(db):
    with sqlite3.connect(db) as con:
        cur = con.cursor()
        result = cur.execute("""
    CREATE TABLE IF NOT EXISTS results(eval_id, question_number,
    realized_prompt, raw_answer, answer, bad_parse, maj_correct, pass_correct)
    """).fetchall()
        con.commit()
    return result

def add_evaluation(db, model, dataset, sub_set, start_time, cot, n_shot, sampler, n_samples, prompt_name, struct_name):
    with sqlite3.connect(db) as con:
        cur = con.cursor()
        cur.execute("""
        INSERT INTO evaluations (model, dataset, sub_set, start_time, cot, n_shot, sampler, n_samples, prompt_name, struct_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [model, dataset, sub_set, start_time,  cot, n_shot, sampler, n_samples, prompt_name, struct_name])
        result = cur.lastrowid
        con.commit()
    return result

def add_result(db, eval_id, question_number, realized_prompt, raw_answer, bad_parse, maj_correct, pass_correct):
    with sqlite3.connect(db) as con:
        cur = con.cursor()
        cur.execute("""
        INSERT INTO results (eval_id, question_number,
        realized_prompt, raw_answer, bad_parse, maj_correct, pass_correct) VALUES
        (?, ?, ?, ?, ?, ?, ?)
        """, [eval_id, question_number, realized_prompt, raw_answer,
               bad_parse, maj_correct, pass_correct])
        result = cur.lastrowid
        con.commit()
    return result
        

def update_evaluation_end(db, eval_id):
    current_time = datetime.now()
    with sqlite3.connect(db) as con:
        cur = con.cursor()
        result = cur.execute(
            """
            UPDATE evaluations SET end_time = ? WHERE rowid = ?
            """, (current_time, eval_id)
        ).fetchall()
        con.commit()
    return result

def eval_results(db, eval_id):
    with sqlite3.connect(db) as con:
        cur = con.cursor()
        result = cur.execute(
            """
            with summary as
            (SELECT eval_id, count(eval_id) as total, avg(maj_correct) as maj_acc, avg(pass_correct) as pass_acc
            from results
            where eval_id = ?
            group by eval_id)
            SELECT model, prompt_name, struct_name, total, maj_acc, pass_acc
            from evaluations
            join summary
            on evaluations.rowid = summary.eval_id;
            """, (eval_id,)
        ).fetchone()
    return result

def display_eval_results(db, eval_id):
    out = "{0}-{1}-{2} {3} obs - MAJ_ACC: {4:0.4f} PASS_ACC: {5:0.4f} "
    result = eval_results(db, eval_id)
    return out.format(*result)

def leaderboard(db, min_obs=0):
    with sqlite3.connect(db) as con:
        cur = con.cursor()
        result = cur.execute(
            """
            with summary as
            (SELECT eval_id, count(eval_id) as total, avg(bad_parse) as pe, 
            avg(maj_correct) as maj_acc, avg(pass_correct) as pass_acc
            from results
            group by eval_id)
            SELECT model, sub_set, prompt_name, struct_name, n_shot, sampler, n_samples, total, maj_acc, pass_acc
            from evaluations
            join summary
            on evaluations.rowid = summary.eval_id
            where total > ?
            order by model, maj_acc desc, pass_acc
            """, (min_obs,)
        ).fetchall()
    return result