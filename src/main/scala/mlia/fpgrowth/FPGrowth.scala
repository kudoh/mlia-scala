package mlia.fpgrowth

import scala.annotation.tailrec

object FPGrowth {

  case class Tree(nodes: Array[TreeNode] = Array(new TreeNode("Null Set", None, None, 1))) {

    def add(parent: TreeNode, newNode: TreeNode) = {
      nodes(nodes.indexOf(parent)).addChild(newNode)
      Tree(nodes :+ newNode)
    }

    override def toString = s"[${nodes.head.disp(0)}]"
  }

  case class TreeNode(name: String,
                      nodeLink: Option[TreeNode] = None,
                      parent: Option[TreeNode] = None) {

    import scala.collection.mutable

    val children: mutable.Map[String, TreeNode] = mutable.Map.empty
    var count = 0

    def this(name: String,
             nodeLink: Option[TreeNode],
             parent: Option[TreeNode],
             count: Int) = {
      this(name, nodeLink, parent)
      this.count = count
    }

    override def toString = s"[$name:$count]"

    def inc() = { count += 1; this }

    def disp(ind: Int): String = "  " * ind + toString + "\n" + children.map(_._2.disp(ind + 1)).mkString("\n")

    def addChild(child: TreeNode) { children += (child.name -> child) }

    def link(node: TreeNode) = copy(nodeLink = Some(node))
  }

  /**
   * Creates FP-tree from dataset but don't mine
   */
  def createTree(dataSet: Map[Set[String], Int],
                 minSup: Int = 1): (Option[Tree], Option[Map[String, Header]]) = {

    // first pass counts frequency of occurrence
    val headerTable = dataSet.foldLeft(Map.empty[String, Header]) {
      (outer, tran) =>
        tran._1.foldLeft(outer) {
          (inner, item) =>
            inner + (item -> inner.getOrElse(item, Header(0, None)).inc)
        }
    }

    val freqItemSet = headerTable.filter(_._2.count >= minSup).keys.toSet

    if (freqItemSet.isEmpty) (None, None)
    else {
      val (tree, updatedHeader) = dataSet.foldLeft(Tree(), headerTable) {
        case ((curTree, table), (tranSet, count)) =>
          val localD = tranSet.foldLeft(Map.empty[String, Int]) {
            case ((curLocalD), item) =>
              if (freqItemSet.contains(item)) curLocalD + (item -> headerTable(item).count) else curLocalD
          }
          if (localD.nonEmpty) {
            val orderedItems = localD.toSeq.sortWith((x1, x2) => (x1._2 compare x2._2) > 0).map(_._1)
            updateTree(orderedItems.toList, curTree, curTree.nodes.head, table, count)
          } else (curTree, table)
      }
      (Some(tree), Some(updatedHeader))
    }
  }

  /**
   * Grows FP-Tree.
   */
  def updateTree(items: List[String],
                 tree: Tree,
                 parent: TreeNode,
                 headerTable: Map[String, Header],
                 count: Int): (Tree, Map[String, Header]) = {

    val (newTree, updateHeader, nextParent) = if (parent.children.contains(items.head)) {
      (tree, headerTable, parent.children(items.head).inc())
    } else {
      val newNode = new TreeNode(items.head, None, Some(parent), count)
      val newT = tree.add(parent, newNode)

      val newHeader = headerTable(items.head).topLink map { link =>
        items.head -> makeHeader(headerTable(items.head), newNode)
      } getOrElse {
        items.head -> (headerTable(items.head) link newNode)
      }
      (newT, headerTable + newHeader, newNode)
    }

    if (items.size > 1) {
      updateTree(items.tail, newTree, nextParent, updateHeader, count)
    } else (newTree, updateHeader)
  }

  def makeHeader(oldHeader: Header, targetNode: TreeNode): Header = {
    @tailrec
    def loop(next: TreeNode, state: TreeNode): TreeNode = next.nodeLink match {
      case None => state
      case Some(x) =>
        loop(state, state link x)
    }
    Header(oldHeader.count, oldHeader.topLink.map(x => loop(x, targetNode)))
  }

  case class Header(count: Int, topLink: Option[TreeNode]) {
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
