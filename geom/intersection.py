# -*- coding: utf-8 -*-
"""
这个文件里面的函数现在已经不用了。

以前可能是想在 Python 里面自己写三角形相交检测，
还有边切分这些操作。但是现在正式流程已经不走这里了。

现在项目里面真正用的是 CGAL 那一套：

1. cgal_bridge/check_self_intersections
   用来检测三角面之间有没有自相交。

2. cgal_bridge/autorefine_obj
   用来让 CGAL 自动处理自相交修复。

所以这个文件只是保留下来，主要是为了说明以前大概想怎么做。
如果有人误调用这里的函数，就直接报错，提醒他去用 CGAL。
"""

from __future__ import annotations


def tri_tri_intersection(*args, **kwargs):
    """
    这个函数以前是想做三角形和三角形的相交检测。

    但是现在不用 Python 自己写这种近似检测了。
    因为三角形相交检测比较容易遇到数值误差问题，
    比如两个三角形刚好贴着、共面、非常接近的时候，
    自己写很容易判断错。

    所以现在统一交给 CGAL 来做。
    """
    raise RuntimeError(
        "geom.intersection.tri_tri_intersection() 现在只是一个保留的旧函数，"
        "正式流程已经不用它了。请使用 cgal_bridge/check_self_intersections "
        "和 cgal_bridge/autorefine_obj。"
    )


def split_edge_if_needed(*args, **kwargs):
    """
    这个函数以前可能是想在 Python 里面手动切边。

    比如两个三角形相交以后，需要在交点位置插入新点，
    然后把原来的边或者三角形重新拆开。

    但是这部分逻辑其实比较复杂，自己写很容易出问题。
    所以现在项目里面把这种自相交修复和边切分的事情，
    都交给 CGAL 的 autorefinement 去处理。
    """
    raise RuntimeError(
        "geom.intersection.split_edge_if_needed() 现在只是一个保留的旧函数，"
        "正式流程已经不用它了。自相交修复中的边插入和三角面切分，"
        "现在交给 CGAL autorefinement 来处理。"
    )