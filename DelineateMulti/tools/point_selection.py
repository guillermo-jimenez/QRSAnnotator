import plotly.graph_objs as go

def create_vertical_lines_tuple(fig, points, fn):
    shapes = []
    if fn in points:
        for p in points[fn]:
            for i in range(len(fig.data)):
                shapes.append(
                    go.layout.Shape(
                        type="line",
                        xref="x" + str(i+1),
                        yref="y" + str(i+1),
                        x0=p,
                        y0=fig.data[0].y.min(),
                        x1=p,
                        y1=fig.data[0].y.max(),
                    )
                )
                # shapes.append(
                #     go.layout.Shape(
                #         type="line",
                #         xref="x" + str(i+1),
                #         yref="y" + str(i+1),
                #         x0=off,
                #         y0=fig.data[0].y.min(),
                #         x1=off,
                #         y1=fig.data[0].y.max(),
                #     )
                # )
    return tuple(shapes)