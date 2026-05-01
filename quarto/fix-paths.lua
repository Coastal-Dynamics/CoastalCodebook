function Image(el)
  -- Replaces "../images/" with "images/" in the rendered HTML
  el.src = el.src:gsub("%.%./images/", "images/")
  return el
end
