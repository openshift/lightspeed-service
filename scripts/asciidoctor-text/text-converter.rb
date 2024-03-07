module Asciidoctor

(QUOTE_TAGS = {
  monospaced:  ['<code>', '</code>', true],
  emphasis:    ['<em>', '</em>', true],
  strong:      ['<strong>', '</strong>', true],
  double:      ['&#8220;', '&#8221;'],
  single:      ['&#8216;', '&#8217;'],
  mark:        ['<mark>', '</mark>', true],
  superscript: ['<sup>', '</sup>', true],
  subscript:   ['<sub>', '</sub>', true],
  asciimath:   ['\$', '\$'],
  latexmath:   ['\(', '\)'],
  # Opal can't resolve these constants when referenced here
  #asciimath:  INLINE_MATH_DELIMITERS[:asciimath] + [false],
  #latexmath:  INLINE_MATH_DELIMITERS[:latexmath] + [false],
}).default = ['', '']

class Converter::TextConverter < Converter::Base
  register_for 'text'

  def initialize *args
    super
    outfilesuffix '.txt'
  end
  def convert node, transform = node.node_name, opts = nil
    case transform
    when 'document', 'section'
      [(decode node.title) + %(\n), (decode node.content)].join
    when 'paragraph'
      (decode node.content.tr ?\n, ' ') << ?\n
    when 'olist', 'ulist', 'colist'
      result = []
      node.items.each do |item|
        result << %(#{(decode item.text)}) << %(\n\n)
      end
      result.join
    when 'dlist'
      result = [%(\n)]
      node.items.each do |terms, dd|
        terms.each do |dt|
          result << %(#{(decode dt.text)}) << ?\n
        end
        if defined?(dd.text)
          result << %(#{(decode dd.text)}) << ?\n
        end
      end
      result.join << ?\n
    else
      (transform.start_with? 'inline_') ? (decode node.text) : (decode node.content)
    end
  end

  def decode str
    unless str.nil?
      str = str.
        gsub('&lt;', '<').
        gsub('&gt;', '>').
        gsub('&#43;', '+').      # plus sign; alternately could use \c(pl
        gsub('&#160;', ' ').    # non-breaking space
        gsub('&#8201;', ' ').    # thin space
        gsub('&#8211;', '-'). # en dash
        gsub('&#8212;', '-'). # em dash
        gsub('&#8216;', %(')). # left single quotation mark
        gsub('&#8217;', %(')). # right single quotation mark
        gsub('&#8220;', %(")). # left double quotation mark
        gsub('&#8221;', %("")). # right double quotation mark
        gsub('&#8592;', '<-'). # leftwards arrow
        gsub('&#8594;', '->'). # rightwards arrow
        gsub('&#8656;', '->'). # leftwards double arrow
        gsub('&#8658;', '<-'). # rightwards double arrow
        gsub('&amp;', '&').      # literal ampersand (NOTE must take place after any other replacement that includes &)
        gsub('\'', %(')).     # apostrophe / neutral single quote
        rstrip                   # strip trailing space
      end
    str
  end

end
end
