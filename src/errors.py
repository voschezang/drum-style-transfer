# error lib?


def typeError(expected, value):
    raise TypeError('expected \n  |> %s \n but found \n  |> %s' %
                    (expected, type(value)))
