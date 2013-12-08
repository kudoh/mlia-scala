package mlia.fpgrowth

import scala.annotation.tailrec

object FPGrowth {

  /**
   * Creates FP-tree from dataset but don't mine
   */
  def createTree(dataSet: Map[Set[String], Int],
                 minSup: Int = 1): (Option[Tree], Option[Map[String, Header]]) = {

    // first pass counts frequency of occurrence
    val headerTable = dataSet.foldLeft(Map.empty[String, Header]) { (outer, tran) =>
      tran._1.foldLeft(outer) { (inner, item) =>
        inner + (item -> inner.getOrElse(item, Header(0, None)).inc)
      }
    }

    val freqItemSet = headerTable.filter(_._2.count >= minSup).keys.toSet

    if (freqItemSet.isEmpty) (None, None)
    else {
      val (tree, updatedHeader) = dataSet.foldLeft(Tree(), headerTable) {
        case ((curTree, table), (tranSet, count)) =>
          val localD = tranSet.foldLeft(Map.empty[String, Int]) { case ((curLocalD), item) =>
            if (freqItemSet.contains(item)) curLocalD + (item -> headerTable(item).count) else curLocalD
          }
          if (localD.nonEmpty) {
            val orderedItems = localD.toSeq.sortWith((x1, x2) => x1._2 > x2._2).map(_._1)
            updateTree(orderedItems.toList, curTree, curTree.nodes.head, table, count)
          } else (curTree, table)
      }
      (Some(tree), Some(updatedHeader))
    }
  }

  /**
   * Grows FP-Tree.
   */
  @tailrec
  def updateTree(items: List[String],
                 tree: Tree,
                 parent: TreeNode,
                 headerTable: Map[String, Header],
                 count: Int): (Tree, Map[String, Header]) = {

    items match {
      case Nil => (tree, headerTable)
      case (x :: xs) =>

        val (newTree, updateHeader, nextParent) =
          if (parent.children.contains(x)) {
            // if parent already has item, just add up.
            (tree, headerTable, parent.children(x).inc())
          } else {
            // create new tree node with given count and update FP-Tree
            val newNode = new TreeNode(x, Some(parent), count)
            val newT = tree.add(parent, newNode)

            // re-create header table node link
            val newHeader = headerTable(x).topLink map { _ =>
              x -> makeHeader(headerTable(x), newNode)
            } getOrElse {
              x -> (headerTable(x) link newNode)
            }
            (newT, headerTable + newHeader, newNode)
          }
        // try next item
        updateTree(xs, newTree, nextParent, updateHeader, count)
    }
  }

  def makeHeader(oldHeader: Header, targetNode: TreeNode): Header = {
    @tailrec
    def last(next: TreeNode): TreeNode = next.nodeLink match {
      case None => next
      case Some(x) => last(x)
    }
    oldHeader.topLink.map { link =>
      last(link) link targetNode
      Header(oldHeader.count, Some(link))
    } getOrElse Header(oldHeader.count, Some(targetNode))
  }

  def ascendTree(leafNode: TreeNode): List[String] = {
    @tailrec
    def loop(curNode: TreeNode, result: List[String]): List[String] = curNode.parent match {
      case None => result
      case Some(p) => loop(p, p.name :: result)
    }
    loop(leafNode, List.empty)
  }

  def findPrefixPath(treeNode: TreeNode): Map[ItemSet, Int] = {
    @tailrec
    def loop(node: TreeNode, result: Map[ItemSet, Int]): Map[ItemSet, Int] = {
      node.nodeLink match {
        case None => result
        case Some(l) =>
          val prefixes = ascendTree(l)
          val newR = if (prefixes.size > 1) result + (ItemSet(prefixes.tail.toSet) -> l.count) else result
          loop(l, newR)
      }
    }
    val top: List[String] = ascendTree(treeNode)
    val initMap = if (top.size > 1) Map(ItemSet(top.tail.toSet) -> treeNode.count) else Map.empty[ItemSet, Int]
    loop(treeNode, initMap)
  }

  def printHeaderTable(t: Map[String, Header]) {
    @tailrec
    def correctLink(node: TreeNode, result: List[TreeNode]): List[TreeNode] = node.nodeLink match {
      case None => result
      case Some(x) => correctLink(x, x :: result)
    }
    t.foreach { case (name, header) =>
      println("-" * 10)
      println(s"item: $name, count: ${header.count}")
      header.topLink.foreach { link =>
        println(correctLink(link, List(link)).mkString(" -> "))
      }
    }
  }

  case class Tree(nodes: Array[TreeNode] = Array(new TreeNode("Null Set", None, 1))) {

    def add(parent: TreeNode, newNode: TreeNode) = {
      nodes(nodes.indexOf(parent)).addChild(newNode)
      Tree(nodes :+ newNode)
    }

    override def toString = s"[${nodes.head.disp(0)}]"
  }

  /**
   * this tree node is mutable.
   */
  case class TreeNode(name: String,
                      parent: Option[TreeNode] = None) {

    import scala.collection.mutable

    val children: mutable.Map[String, TreeNode] = mutable.Map.empty
    var nodeLink: Option[TreeNode] = None
    var count = 0

    def this(name: String,
             parent: Option[TreeNode],
             count: Int) = {
      this(name, parent)
      this.count = count
    }

    override def toString = s"[$name:$count]"

    def inc() = { count += 1; this }

    def disp(ind: Int): String = "  " * ind + toString + "\n" + children.map(_._2.disp(ind + 1)).mkString("\n")

    def addChild(child: TreeNode) = { children += (child.name -> child); this }

    def link(node: TreeNode) = { nodeLink = Some(node); this }
  }

  case class Header(count: Int, topLink: Option[TreeNode]) {

    def top: TreeNode = topLink.getOrElse[TreeNode](throw new IllegalStateException("top link is None."))

    def inc = Header(count + 1, topLink)

    def link(node: TreeNode) = Header(count, Some(node))
  }

  object ItemSet {
    def apply(x: String): ItemSet = new ItemSet(Set(x))

    def apply(arr: Array[String]): ItemSet = new ItemSet(arr.toSet)

    implicit def set2ItemSet(x: Set[String]): ItemSet = new ItemSet(x)
  }

  case class ItemSet(x: Set[String]) extends Set[String] {

    override def toString() = s"Items[${x.mkString(", ")}]"

    def contains(elem: String): Boolean = x.contains(elem)

    def +(elem: String): Set[String] = x + elem

    def -(elem: String): Set[String] = x - elem

    def iterator: Iterator[String] = x.iterator
  }

}
