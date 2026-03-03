from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def ascii_ignore(x):
  if x:
      return x.encode('ascii', 'ignore').decode('ascii').strip().replace('?', '').replace(')', '') \
                .replace('(', '').replace('%', '').replace('=', '').replace(':', '').replace('&', '') \
                .replace('*', '').replace('"', '').replace("'", '').replace(">", '').replace("<", '') \
                .replace("\\", '').replace("}", '').replace("{", '').replace("$", '').replace("@", '') \
                .replace("+", '').replace("/", '').replace("]", '').replace("[", '').replace(";", '') \
                .replace("~", '').replace("#", '').replace(",", '')
  else:
      return None
