import os
import jsonpickle
import json
import functools
import inspect
import sqlite3


def cache_in_file(wrapped_function):
  """
  Wraps a function to enable caching its results in a file.

  - If a keyword argument 'cache_key' is provided, the function checks if a cached result exists in a file with the name pattern "cache_<wrapped method name>_<cache_key>.json".
  - If the cached file exists, the result is read from the file and returned.
  - If the cached file does not exist, the wrapped function is called, its result is saved to the cache file, and the result is returned.
  - If the 'cache_key' argument is not provided, the wrapper simply delegates to the wrapped function and returns its result.

  :param wrapped_function: Function to be wrapped.
  :return: Wrapped function with caching behavior.
  """

  @functools.wraps(wrapped_function)
  def wrapper(*args, **kwargs):
    cache_key = kwargs.pop('cache_key', None)
    if not cache_key:
      return wrapped_function(*args, **kwargs)

    cache_file_name = f'cache_{wrapped_function.__name__}_{cache_key}.json'
    try:
      if os.path.exists(cache_file_name):
        with open(cache_file_name, 'r', encoding='utf-8') as cache_file:
          cached_result = jsonpickle.decode(cache_file.read())
          return cached_result
    except Exception as e:
      print(f"An error occurred while reading from cache: {e}")
      # Fall through to compute the actual method in case of an error

    # Compute the result if no cache was found or an error occurred
    result = wrapped_function(*args, **kwargs)
    try:
      with open(cache_file_name, 'w', encoding='utf-8') as cache_file:
        cache_file.write(jsonpickle.encode(result))
    except Exception as e:
      print(
        f"An error occurred while writing to cache for {wrapped_function.__name__}() and key {cache_key}: {e}"
      )

    return result

  return wrapper

class CacheTwoStringResults:
  """
  This class wraps a method that accepts two string arguments and
  caches the results in cache.db (SQLite).

  The wrapped method should be commutative and the first two arguments are always sorted
  alphabetically for retrieval of the cached results and delegated calls.

  The caching database connection is cached within this class, and a new connection
  will be created if the existing one fails.

  Attributes:
    wrapped_method (function): The method to be wrapped, which accepts two string arguments.
    connection: A cached SQLite connection.
    created_tables (set): A set that stores the names of already created tables.
  """

  def __init__(self, wrapped_method):
    sig = inspect.signature(wrapped_method)
    params = list(sig.parameters.values())
    if len(params) < 2 or params[0].annotation != str or params[1].annotation != str:
      raise ValueError("The first two arguments of the wrapped method must be strings.")
    self.wrapped_method = wrapped_method
    self.connection = None
    self.created_tables = set()
    functools.update_wrapper(self, wrapped_method)

  def get_connection(self):
    if not self.connection:
      try:
        self.connection = sqlite3.connect("cache.db")
      except sqlite3.Error as e:
        print(f"SQLite error occurred while connecting to cache.db: {e}")
        self.connection = None
    return self.connection

  def close_connection(self):
    if self.connection is not None:
      self.connection.close()
      self.connection = None

  def create_table_if_not_exists(self, conn, table_name):
    if table_name not in self.created_tables:
      cursor = conn.cursor()
      cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} " \
                     "(arg1 TEXT, arg2 TEXT, value TEXT, PRIMARY KEY(arg1, arg2))")
      self.created_tables.add(table_name)

  def __call__(self, *args, **kwargs):
    # Sort first two arguments alphabetically
    arg1, arg2 = sorted(args[:2])
    args = (arg1, arg2) + args[2:]
    
    table_name = self.wrapped_method.__name__

    try:
      conn = self.get_connection()
      if conn is None:
        raise sqlite3.Error("Unable to establish connection to cache.db")

      self.create_table_if_not_exists(conn, table_name)

      cursor = conn.cursor()
      cursor.execute(f"SELECT value FROM {table_name} WHERE arg1 = ? AND arg2 = ?", (arg1, arg2))
      row = cursor.fetchone()

      if row:
        return json.loads(row[0])

      result = self.wrapped_method(*args, **kwargs)
      cursor.execute(
        f"INSERT INTO {table_name} (arg1, arg2, value) VALUES (?, ?, ?)",
        (arg1, arg2, json.dumps(result)))
      conn.commit()
      return result

    except sqlite3.Error as e:
      print(f"SQLite error occurred: {e}")
      # Recycle the connection
      self.close_connection()
      return self.wrapped_method(*args, **kwargs)
      

def cache_two_string_results(wrapped_method):
  return CacheTwoStringResults(wrapped_method)