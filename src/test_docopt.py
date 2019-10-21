"""Overview:
  サブコマンド・オプション・引数をdocoptで受け、keyとvalueを全表示する

Usage:
    test_docopt [options <op>]

Options:
    --aaa <op> : aaa bbb ccc
    --bbb <op> : aaa bbb ccc
"""

from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__)
    print("  {0:<20}{1:<20}{2:<20}".format("kye", "value", "type"))
    print("  {0:-<60}".format(""))
    for k,v in args.items():
        print("  {0:<20}{1:<20}{2:<20}".format(str(k), str(v), str(type(v))))
