# error lib?


def typeError(expected, value):
    raise TypeError('expected \n  |> %s \n but found \n  |> %s' %
                    ('np.array', type(value)))
