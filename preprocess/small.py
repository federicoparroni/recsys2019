def create(filename, N=1000, folder='dataset', insert_index_col=False):
  inp = '{}/original/{}.csv'.format(folder, filename)
  dest = '{}/preprocessed/{}_small.csv'.format(folder, filename)

  with open(inp, 'r') as file:
    with open(dest, 'w+') as out:
      # header
      line = file.readline()
      out.write(line)

      # data
      for i in range(N):
        line = file.readline()
        if line is not None:
          if insert_index_col:
            out.write('{},'.format(i))
          out.write(line)

          print('{}/{}'.format(i+1, N), end='\r')
        else:
          break

if __name__ == "__main__":
  create('train')
  create('test')
