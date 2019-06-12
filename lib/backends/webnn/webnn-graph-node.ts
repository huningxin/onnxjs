import {Attribute} from '../../attribute';
import {Graph} from '../../graph';

interface Dictionary<T> {
  [key: string]: T;
}

export class WebNNGraphNode implements Graph.Node {
  constructor(public nodes: ReadonlyArray<Graph.Node>, public inputs: number[], public outputs: number[]) {
    this.name = `${this.graphSummary()} (${this.hashCode()})`;
    this.opType = 'WebNNGraph';
    this.attributes = new Attribute(null);
    this.executeNode = true;
  }

  name: string;
  opType: string;
  attributes: Attribute;
  executeNode: boolean;

  graphSummary() {
    const objectEntries = (o: Dictionary<number>) => Object.keys(o).map(k => [k, o[k]]);  // polyfill for Object.entries
    return objectEntries(this.nodes.map((node) => node.opType).reduce((cnt: Dictionary<number>, t: string) => {
             cnt[t] ? cnt[t]++ : cnt[t] = 1;
             return cnt;
           }, {})).map((n: Array<string|number>) => `${n[0]} x ${n[1]}`).join(', ');
  }

  hashCode() {
    return (Array.from(JSON.stringify(this)).reduce((s, c) => Math.imul(31, s) + c.charCodeAt(0) | 0, 0) + 2 ** 31)
        .toString(16);
  }
}
