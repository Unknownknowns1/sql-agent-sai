import os
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SQL Manager Pro",
    page_icon="🗄️",
    layout="wide",
)


required_db_vars = {
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_NAME": DB_NAME,
}

missing_db_vars = [name for name, value in required_db_vars.items() if not value]
if missing_db_vars:
    st.error("❌ Missing MySQL environment variables in .env")
    st.code(
        "Add the following values:\n"
        "DB_USER=your_user\n"
        "DB_PASSWORD=your_password\n"
        "DB_HOST=localhost\n"
        "DB_PORT=3306\n"
        "DB_NAME=your_database\n\n"
        f"Missing: {', '.join(missing_db_vars)}"
    )
    st.stop()


# =========================================================
# DATABASE CONNECTION
# =========================================================

password = quote_plus(DB_PASSWORD)
MYSQL_URI = (
    f"mysql+pymysql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

try:
    engine = create_engine(MYSQL_URI)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
except Exception as e:
    st.error("❌ MySQL connection failed. Check your DB credentials and server.")
    st.code(
        f"DB_USER={DB_USER}\n"
        f"DB_HOST={DB_HOST}\n"
        f"DB_PORT={DB_PORT}\n"
        f"DB_NAME={DB_NAME}\n\n"
        f"{type(e).__name__}: {e}"
    )
    st.stop()


# =========================================================
# DATABASE HELPERS
# =========================================================

def quote_identifier(identifier: str) -> str:
    return f"`{str(identifier).replace('`', '``')}`"


def get_table_list() -> list[str]:
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT TABLE_NAME
                FROM information_schema.tables
                WHERE table_schema = :db_name
                ORDER BY TABLE_NAME
                """
            ),
            {"db_name": DB_NAME},
        )
        return [row[0] for row in result.fetchall()]


def get_table_columns(table_name: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, EXTRA
                FROM information_schema.columns
                WHERE table_schema = :db_name
                  AND table_name = :table_name
                ORDER BY ORDINAL_POSITION
                """
            ),
            conn,
            params={"db_name": DB_NAME, "table_name": table_name},
        )


def get_primary_key(table_name: str) -> str:
    cols = get_table_columns(table_name)
    if cols.empty:
        return ""
    pk = cols[cols["COLUMN_KEY"] == "PRI"]
    if not pk.empty:
        return str(pk.iloc[0]["COLUMN_NAME"])
    return str(cols.iloc[0]["COLUMN_NAME"])


def fetch_table_data(table_name: str, limit: int = 200) -> pd.DataFrame:
    table_ref = quote_identifier(table_name)
    query = f"SELECT * FROM {table_ref} LIMIT :limit"
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params={"limit": limit})


def insert_row(table_name: str, row_data: dict) -> None:
    cols = list(row_data.keys())
    sql = (
        f"INSERT INTO {quote_identifier(table_name)} "
        f"({', '.join(quote_identifier(col) for col in cols)}) "
        f"VALUES ({', '.join(f':{col}' for col in cols)})"
    )
    with engine.begin() as conn:
        conn.execute(text(sql), row_data)


def update_row(table_name: str, pk_column: str, pk_value, updates: dict) -> None:
    if not updates:
        return
    assignments = ", ".join(
        f"{quote_identifier(col)} = :{col}" for col in updates.keys()
    )
    sql = (
        f"UPDATE {quote_identifier(table_name)} "
        f"SET {assignments} "
        f"WHERE {quote_identifier(pk_column)} = :pk_value"
    )
    params = {**updates, "pk_value": pk_value}
    with engine.begin() as conn:
        conn.execute(text(sql), params)


def delete_row(table_name: str, pk_column: str, pk_value) -> None:
    sql = (
        f"DELETE FROM {quote_identifier(table_name)} "
        f"WHERE {quote_identifier(pk_column)} = :pk_value"
    )
    with engine.begin() as conn:
        conn.execute(text(sql), {"pk_value": pk_value})


def run_sql_command(sql: str):
    normalized = sql.strip()
    if not normalized:
        return None, "No SQL provided."

    with engine.connect() as conn:
        if normalized.lower().startswith("select"):
            df = pd.read_sql(text(normalized), conn)
            return df, "Query executed successfully."

        result = conn.execute(text(normalized))
        conn.commit()

        if result.returns_rows:
            rows = result.fetchall()
            columns = result.keys()
            return pd.DataFrame(rows, columns=list(columns)), "Query executed successfully."
        return None, "Query executed successfully."


# =========================================================
# LANGCHAIN DATABASE
# =========================================================

try:
    db = SQLDatabase.from_uri(MYSQL_URI)
except Exception as e:
    st.error("❌ Could not inspect the database schema.")
    st.code(str(e))
    st.stop()


# =========================================================
# GROQ
# =========================================================

if GROQ_API_KEY:
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=GROQ_API_KEY,
    )

    custom_sql_prefix = """You are a database assistant that can read and modify the database.
You may generate SQL for SELECT, INSERT, UPDATE, DELETE, and ALTER TABLE when the user asks to add, edit, delete, or change data.
Always validate the SQL before running it, keep queries relevant, and use the most specific table/columns needed.
If the task is not related to the database, respond with "I don't know".
A user request to add, update, insert, or remove records is valid and should be executed when the database schema allows it.

Dialects: {dialect}
Top results limit: {top_k}
"""

    agent = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        agent_type="tool-calling",
        prefix=custom_sql_prefix,
        top_k=20,
    )
else:
    llm = None
    agent = None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🗄️ SQL Manager Pro")

menu = st.sidebar.radio(
    "Choose Option",
    [
        "🏠 Dashboard",
        "➕ Add Employee",
        "👥 View Employees",
        "🧰 SQL Manager",
        "🤖 SQL AI Agent",
    ],
)


# =========================================================
# DASHBOARD
# =========================================================

if menu == "🏠 Dashboard":
    st.title("📊 Employee Dashboard")

    try:
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM employees")).scalar()
            avg_salary = conn.execute(text("SELECT AVG(salary) FROM employees")).scalar()
            max_salary = conn.execute(text("SELECT MAX(salary) FROM employees")).scalar()
            min_salary = conn.execute(text("SELECT MIN(salary) FROM employees")).scalar()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Employees", total)
        col2.metric("💰 Average Salary", f"₹{avg_salary:,.0f}" if avg_salary else "₹0")
        col3.metric("⬆️ Highest Salary", f"₹{max_salary:,.0f}" if max_salary else "₹0")
        col4.metric("⬇️ Lowest Salary", f"₹{min_salary:,.0f}" if min_salary else "₹0")

        st.divider()
        st.subheader("📋 Recent Employees")

        df = pd.read_sql(
            """
            SELECT id, name, dept_id, gender, city, salary, manager_id
            FROM employees
            ORDER BY id DESC
            LIMIT 10
            """,
            engine,
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("❌ Dashboard could not load.")
        st.code(str(e))


# =========================================================
# ADD EMPLOYEE
# =========================================================

elif menu == "➕ Add Employee":
    st.title("➕ Add New Employee")

    with st.form("employee_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Employee Name")
            dept_id = st.number_input("Department ID", min_value=1, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        with col2:
            city = st.text_input("City")
            salary = st.number_input("Salary", min_value=0.0, step=1000.0)
            manager_id = st.number_input("Manager ID", min_value=0, step=1, value=0)

        submitted = st.form_submit_button("➕ Add Employee", use_container_width=True)

    if submitted:
        if not name.strip():
            st.warning("Please enter employee name.")
        elif not city.strip():
            st.warning("Please enter city.")
        else:
            try:
                manager_value = None if manager_id == 0 else manager_id
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            """
                            INSERT INTO employees
                            (name, dept_id, gender, city, salary, manager_id)
                            VALUES (:name, :dept_id, :gender, :city, :salary, :manager_id)
                            """
                        ),
                        {
                            "name": name,
                            "dept_id": dept_id,
                            "gender": gender,
                            "city": city,
                            "salary": salary,
                            "manager_id": manager_value,
                        },
                    )
                st.success(f"✅ Employee '{name}' added successfully!")
            except Exception as e:
                st.error("❌ Could not add employee.")
                st.code(str(e))


# =========================================================
# VIEW EMPLOYEES
# =========================================================

elif menu == "👥 View Employees":
    st.title("👥 Employee Records")

    try:
        df = pd.read_sql(
            """
            SELECT id, name, dept_id, gender, city, salary, manager_id
            FROM employees
            ORDER BY id
            """,
            engine,
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.write(f"Total Employees: **{len(df)}**")
    except Exception as e:
        st.error("❌ Unable to load employees.")
        st.code(str(e))


# =========================================================
# SQL MANAGER
# =========================================================

elif menu == "🧰 SQL Manager":
    st.title("🧰 SQL Table Manager")
    st.caption("Add rows, edit records, delete rows, and run custom SQL.")

    tables = get_table_list()
    if not tables:
        st.info("No tables found in this database.")
        st.stop()

    selected_table = st.selectbox("Choose a table", tables)
    columns = get_table_columns(selected_table)

    if columns.empty:
        st.warning(f"The table '{selected_table}' has no columns.")
        st.stop()

    st.subheader(f"Table: {selected_table}")
    df = fetch_table_data(selected_table, limit=200)
    st.dataframe(df, use_container_width=True, hide_index=True)

    pk_column = get_primary_key(selected_table)
    if pk_column:
        st.caption(f"Primary key detected: {pk_column}")

    st.divider()

    st.subheader("➕ Add a row")
    add_form = st.form(f"add_row_{selected_table}", clear_on_submit=True)
    input_values = {}
    for _, row in columns.iterrows():
        col_name = str(row["COLUMN_NAME"])
        if pk_column and col_name == pk_column and "auto_increment" in str(row["EXTRA"]).lower():
            continue

        data_type = str(row["DATA_TYPE"]).lower()
        is_nullable = row["IS_NULLABLE"] == "YES"

        if "int" in data_type or "decimal" in data_type or "float" in data_type or "double" in data_type:
            value = add_form.number_input(
                col_name,
                value=0.0 if "float" in data_type or "double" in data_type or "decimal" in data_type else 0,
                step=1 if "int" in data_type else 0.1,
                key=f"add_{selected_table}_{col_name}",
            )
        else:
            value = add_form.text_input(
                col_name,
                value="" if is_nullable else "",
                key=f"add_{selected_table}_{col_name}",
            )
        input_values[col_name] = value

    if add_form.form_submit_button("Insert row"):
        try:
            filtered_values = {k: v for k, v in input_values.items() if v not in (None, "")}
            if not filtered_values:
                st.warning("No values were supplied for the new row.")
            else:
                insert_row(selected_table, filtered_values)
                st.success(f"✅ Row added to {selected_table}.")
                st.rerun()
        except Exception as e:
            st.error("❌ Insert failed.")
            st.code(str(e))

    st.divider()

    st.subheader("✏️ Update rows")
    edited_df = st.data_editor(
        df,
        key=f"editor_{selected_table}",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
    )

    if st.button("💾 Save table edits"):
        try:
            original_rows = df.to_dict("records")
            edited_rows = edited_df.to_dict("records")

            for idx, row in enumerate(edited_rows):
                if idx >= len(original_rows):
                    insert_row(selected_table, {k: v for k, v in row.items() if not pd.isna(v) and v is not None})
                    continue

                original_row = original_rows[idx]
                changes = {}
                for col in df.columns:
                    new_value = row.get(col)
                    old_value = original_row.get(col)
                    if pd.isna(new_value) and old_value is not None:
                        changes[col] = None
                    elif new_value != old_value:
                        changes[col] = new_value

                if pk_column not in changes and not changes:
                    continue

                if pk_column in changes:
                    del changes[pk_column]

                if changes:
                    update_row(selected_table, pk_column, row.get(pk_column), changes)

            st.success(f"✅ Changes saved for {selected_table}.")
            st.rerun()
        except Exception as e:
            st.error("❌ Update failed.")
            st.code(str(e))

    st.divider()

    st.subheader("🗑️ Delete a row")
    if not df.empty:
        delete_index = st.number_input(
            "Select row index to delete",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
        )
        if st.button("Delete selected row"):
            try:
                row = df.iloc[delete_index]
                pk_value = row[pk_column] if pk_column and pk_column in row else None
                if pk_value is None:
                    st.warning("No primary key found for this table. Delete cannot be performed automatically.")
                else:
                    delete_row(selected_table, pk_column, pk_value)
                    st.success("✅ Row deleted.")
                    st.rerun()
            except Exception as e:
                st.error("❌ Delete failed.")
                st.code(str(e))
    else:
        st.info("This table has no rows to delete.")

    st.divider()

    st.subheader("🧪 SQL Console")
    sql_input = st.text_area(
        "Run custom SQL",
        height=180,
        value=f"SELECT * FROM {quote_identifier(selected_table)} LIMIT 20;",
    )
    if st.button("Execute SQL"):
        try:
            result_df, message = run_sql_command(sql_input)
            if result_df is not None:
                st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.success(message)
        except Exception as e:
            st.error("❌ SQL execution failed.")
            st.code(str(e))


# =========================================================
# AI SQL AGENT
# =========================================================

elif menu == "🤖 SQL AI Agent":
    st.title("🤖 AI SQL Assistant")
    st.write("Ask questions about your database using natural language.")

    if agent is None:
        st.warning("⚠️ GROQ_API_KEY is missing from .env. Add it to enable the AI SQL assistant.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask something about your database...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        response = agent.invoke({"input": question})
                        answer = response["output"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.expander("🔍 Agent Details"):
                            st.write(response)
                    except Exception as e:
                        st.error("❌ AI Agent Error")
                        st.code(str(e))