import React from 'react';

function InlineMarkdown({ text }) {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith('`') && part.endsWith('`')) return <code key={index}>{part.slice(1, -1)}</code>;
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith('*') && part.endsWith('*')) return <em key={index}>{part.slice(1, -1)}</em>;
    return part;
  });
}

function parseTable(lines) {
  const cells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(cell => cell.trim());
  const header = cells(lines[0]);
  const rows = lines.slice(2).map(cells);
  return (
    <div className="ai-markdown-table-wrap">
      <table>
        <thead><tr>{header.map((cell, index) => <th key={index}><InlineMarkdown text={cell} /></th>)}</tr></thead>
        <tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{header.map((_, index) => <td key={index}><InlineMarkdown text={row[index] || ''} /></td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

export default function MarkdownMessage({ content }) {
  const lines = String(content || '').split('\n');
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (line.startsWith('```')) {
      const language = line.slice(3).trim();
      const code = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith('```')) code.push(lines[index++]);
      if (index < lines.length) index += 1;
      blocks.push(<pre key={`code-${index}`}><code className={language ? `language-${language}` : undefined}>{code.join('\n')}</code></pre>);
      continue;
    }
    if (line.trim().startsWith('|') && index + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[index + 1])) {
      const table = [line, lines[index + 1]];
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith('|')) table.push(lines[index++]);
      blocks.push(React.cloneElement(parseTable(table), { key: `table-${index}` }));
      continue;
    }
    if (!line.trim()) { index += 1; continue; }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const Tag = `h${heading[1].length}`;
      blocks.push(<Tag key={`heading-${index}`}><InlineMarkdown text={heading[2]} /></Tag>);
      index += 1;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) items.push(lines[index++].replace(/^\s*[-*]\s+/, ''));
      blocks.push(<ul key={`list-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}><InlineMarkdown text={item} /></li>)}</ul>);
      continue;
    }
    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !lines[index].startsWith('```') && !/^(#{1,3})\s+/.test(lines[index]) && !/^\s*[-*]\s+/.test(lines[index])) paragraph.push(lines[index++]);
    blocks.push(<p key={`paragraph-${index}`}>{paragraph.map((part, partIndex) => <React.Fragment key={partIndex}>{partIndex > 0 && <br />}<InlineMarkdown text={part} /></React.Fragment>)}</p>);
  }
  return <div className="ai-markdown">{blocks}</div>;
}
