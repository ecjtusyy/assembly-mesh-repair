from __future__ import annotations


def retriangulate_polygon(*args, **kwargs):
    """
    这个函数以前可能是想做多边形重新三角剖分。

    比如一个多边形有很多个点：

        p1, p2, p3, p4, p5

    那就需要把它拆成多个三角形。

    但是在网格修复里面，这件事并不简单。
    因为三角面相交以后，新增点在哪里、边怎么切、面怎么重新连接，
    都会影响最后网格是不是合法。

    所以现在不在 Python 里面手写这部分逻辑了，
    而是交给 CGAL 的 autorefine_triangle_soup() 来处理。
    """
    raise RuntimeError(
        "geom.retriangle.retriangulate_polygon() 现在只是一个保留的旧函数，"
        "正式修复流程已经不用它了。三角面切分和重新剖分现在由 "
        "CGAL autorefine_triangle_soup() 处理。"
    )